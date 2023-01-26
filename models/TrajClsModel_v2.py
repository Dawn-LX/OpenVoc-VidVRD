import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



def _to_predict_cls_ids(
    cls_split,
    num_base,
    num_novel,
    cls_probs,
):
    _,num_cls = cls_probs.shape
    scores,cls_ids = torch.max(cls_probs,dim=-1)  # (n_det,)

    # 0,[1,2,....,49,50],[51,52,...,79,80]

    if cls_split == "base":
        assert num_cls == num_base  # for object class in VidOR, num_base == 50
        cls_ids += 1  # 0 ~ 49 --> 1 ~ 50,  len == 50
        
    elif cls_split == "novel":
        assert num_cls == num_novel # 30
        cls_ids += 1 + num_base    # range: 0 ~ 29  --> 51 ~ 80

    elif cls_split == "all":
        assert num_cls == num_base + num_novel  # 80
        cls_ids += 1
        # rang: 0 ~ 79 --> 1 ~ 80
    else:
        assert False, "eval_split must be base, novel, or all"

    
    return scores,cls_ids



class OpenVocTrajCls(nn.Module):
    '''
    use 1/C_B to do negative proposal w.r.t backgorund classification
    '''
    def __init__(
        self,configs,is_train=True,use_distillation=True
    ):
        super().__init__()
        self.is_train = is_train
        self.use_distillation = use_distillation
        self.dim_emb = configs["dim_emb"]
        self.dim_roi = configs["dim_roi"]  # bbox roi feature, refer to Faster-RCNN
        self.dim_hid = configs.get("dim_hidden",self.dim_emb)
        self.num_base = configs["num_base"]
        self.num_novel = configs["num_novel"]
        self.text_emb_path = configs["text_emb_path"]  # prepared_data/vidor_ObjectTextEmbeddings.pth
        self.loss_factor = configs["loss_factor"]
        self.temp_init = configs["temperature_init"]

        text_embeddings = torch.load(self.text_emb_path).float() # float32 # __background__ is all 0's shape == (81,dim_emb)
        text_embeddings_wo_bg = text_embeddings[1:,:]  # (80,dim_emb) we follow the `1/C_B` paper
        self.class_embeddings = nn.Parameter(text_embeddings_wo_bg,requires_grad=False)  # (80,dim_emb)
        # text_embeddings has been normalized in Alpro-text
        
        self.V2L_projection = nn.Sequential(
            nn.Linear(self.dim_roi,self.dim_hid),
            nn.ReLU(),
            nn.Linear(self.dim_hid,self.dim_hid),
            nn.ReLU(),
            nn.Linear(self.dim_hid,self.dim_emb)   # distillation_targets 正负都有
        )
        self.temperature = nn.Parameter(torch.ones([]) * self.temp_init)

        if self.is_train:
            self.reset_classifier_weights("base") # TODO
        else:
            pass
            # call `reset_classifier_weights` externally (before call self.forward_inference) to evaluate on other split 
            # e.g., evaluate on "novel"
            # self.reset_classifier_weights("novel") or "base" or "all"

    def forward(self,batch_data):
        '''
        seg_tag, traj_info,traj_feat,traj_embd, gt_anno, labels = batch_data[0]
        '''
        batch_traj_feats,batch_distil_tgt,batch_labels = batch_data
        
        # if self.use_distillation: TODO
        batch_traj_feats = torch.cat(batch_traj_feats,dim=0)    # (N_traj,2048)
        batch_distil_tgt = torch.cat(batch_distil_tgt,dim=0)    # (N_traj,256) 
        batch_labels = torch.cat(batch_labels,dim=0)            # (N_traj,)  range 0~50

        batch_traj_embs = self.V2L_projection(batch_traj_feats) # (N_traj, 256)
        batch_traj_embs_norm = F.normalize(batch_traj_embs,dim=-1)
        logits = torch.matmul(batch_traj_embs_norm,self.classifier_weights.t()) / self.temperature
        # (N_traj, dim_emb) x (dim_emb, num_base) --> (N_traj,num_base)

        pos_mask = (0< batch_labels) & (batch_labels <= self.num_base)  # 0 stands for __background__, base clsid: 1 ~ 50
        pos_labels = batch_labels[pos_mask] -1   # (n_pos,) # here convert the label range to 0 ~ num_base-1, to fit the CE loss input (0 stands for the first base class)
        pos_logits = logits[pos_mask,:]  # (n_pos,num_base)


        pos_cls_loss = F.cross_entropy(pos_logits,pos_labels,reduction='none')
        # pos_cls_loss equals the following code:
        # pos_cls_loss2 = (-1 * F.log_softmax(pos_logits,dim=-1)*pos_onehot).sum(dim=-1) # pos_onehot.shape == pos_logits.shape
        # assert torch.all(pos_cls_loss == pos_cls_loss2)

        neg_mask  = ~pos_mask
        neg_logits = logits[neg_mask,:]  # (n_neg,num_base)
        neg_target = torch.ones_like(neg_logits) / self.num_base
        neg_cls_loss = (-1 * F.log_softmax(neg_logits,dim=-1)*neg_target).sum(dim=-1)
        ### refer to `test_API/test_CE_loss.py`

        # deal with empty case (apply .mean() to empty tensor will get `nan`)
        d_ = logits.device
        if pos_cls_loss.numel() == 0:  
            pos_cls_loss = torch.zeros(size=(),device=d_)
        if neg_cls_loss.numel() == 0:
            pos_cls_loss = torch.zeros(size=(),device=d_)
        pos_cls_loss = pos_cls_loss.mean() * self.loss_factor["pos_cls"]
        neg_cls_loss = neg_cls_loss.mean() * self.loss_factor["neg_cls"]
        
        loss_dict = {"pos_cls":pos_cls_loss,"neg_cls":neg_cls_loss}

        if self.use_distillation:

            distil_loss = F.l1_loss(batch_traj_embs,batch_distil_tgt,reduction='mean')
            distil_loss *= self.loss_factor["distillation"]
            loss_dict.update({"distil_loss":distil_loss})

        total_loss = torch.stack(list(loss_dict.values())).sum()    # scalar tensor
        loss_for_show = {k:v.detach() for k,v in loss_dict.items()}
        loss_for_show.update({"total":total_loss.detach()})

        return  total_loss, loss_for_show


    def reset_classifier_weights(self,split):
        if split == "base":
            classifier_weights = self.class_embeddings[:self.num_base,:]   # (num_base, dim_emb) == (50,dim_emb) (0,1,2,...,49)  0 index the first base class 
        elif split == "novel":
            classifier_weights =self.class_embeddings[self.num_base:,:]    # (num_novel, dim_emb) == (30,dim_emb)  (50,51,52,...,79)
        elif split == "all":
            classifier_weights = self.class_embeddings                     # (80, dim_emb)   exclude __background__, 0 ~ 79
        else:
            assert False, "split must be base, novel, or all"
        
        self.classifier_weights = classifier_weights
        self.cls_split = split

    @torch.no_grad()
    def forward_inference_bsz1(self,traj_feats_or_embds,input_emb=False):
        if input_emb:  # for teacher model
            traj_embs = traj_feats_or_embds
        else:
            traj_features = traj_feats_or_embds
            traj_embs = self.V2L_projection(traj_features) # (n_det, dim_emb)
        
        traj_embs = F.normalize(traj_embs,dim=-1)
        cls_logits = torch.matmul(traj_embs,self.classifier_weights.t()) / self.temperature # (n_det, num_cls)  # exclude __background__
        cls_probs = torch.softmax(cls_logits,dim=-1)  # (n_det, num_cls) , exclude __background__
        
        scores,cls_ids = _to_predict_cls_ids(
            self.cls_split,
            self.num_base,
            self.num_novel,
            cls_probs
        )
        
        
        return scores,cls_ids
    
    def forward_inference(self,batch_traj_features):

        traj_embs = self.V2L_projection(batch_traj_features) # (N_det, dim_emb)
        traj_embs = F.normalize(traj_embs,dim=-1)
        cls_logits = torch.matmul(traj_embs,self.classifier_weights.t()) / self.temperature # (N_det, num_cls)  # exclude __background__
        cls_probs = torch.softmax(cls_logits,dim=-1)  # (N_det, num_cls) , exclude __background__
        
        scores,cls_ids = _to_predict_cls_ids(
            self.cls_split,
            self.num_base,
            self.num_novel,
            cls_probs
        )
        
        
        return scores,cls_ids

        


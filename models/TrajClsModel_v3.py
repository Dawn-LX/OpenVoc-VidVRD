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
        self.text_emb_path = configs["text_emb_path"]
        self.loss_factor = configs["loss_factor"]

        text_embeddings = torch.load(self.text_emb_path)   # __background__ is all 0's shape == (36,dim_emb)
        self.class_embeddings = nn.Parameter(text_embeddings,requires_grad=False)  # (36,dim_emb)
        
        self.V2L_projection = nn.Sequential(
            nn.Linear(self.dim_roi,self.dim_hid),
            nn.ReLU(),
            nn.Linear(self.dim_hid,self.dim_hid),
            nn.ReLU(),
            nn.Linear(self.dim_hid,self.dim_emb)   # distillation_targets 正负都有
        )
        
        if self.is_train:
            self.reset_classifier_weights("base")
        else:
            self.reset_classifier_weights("novel")
            # This func can be called externally (before call self.forward_inference) to evaluate on other split 
            # (e.g., evaluate on "base" or "all")


    def forward(self,batch_data):
        batch_traj_feats,batch_distil_tgt,batch_labels = batch_data
        
        batch_traj_feats = torch.cat(batch_traj_feats,dim=0)    # (N_traj,2048)
        batch_distil_tgt = torch.cat(batch_distil_tgt,dim=0)    # (N_traj,256) 
        batch_labels = torch.cat(batch_labels,dim=0)            # (N_traj,)  range 0~50
        
        batch_traj_embs = self.V2L_projection(batch_traj_feats) # (N_traj, 256)
        logits = torch.matmul(batch_traj_embs,self.classifier_weights.t())
        # (N_traj, dim_emb) x (dim_emb, num_base) --> (N_traj,num_base)

        pos_mask = (0< batch_labels) & (batch_labels <= self.num_base)  # 0 stands for __background__, base clsid: 1 ~ 50

        cls_loss = F.cross_entropy(logits,batch_labels,reduction='none')  # (N_traj,num_base)
        cls_pos = cls_loss[pos_mask]
        cls_neg = cls_loss[~pos_mask]

        d_ = logits.device
        if cls_pos.numel() == 0:  
            cls_pos = torch.zeros(size=(),device=d_)
        if cls_neg.numel() == 0:
            cls_neg = torch.zeros(size=(),device=d_)
        cls_pos = cls_pos.mean() * self.loss_factor["pos_cls"]
        cls_neg = cls_neg.mean() * self.loss_factor["neg_cls"]

        loss_dict = {"pos_cls":cls_pos,"neg_cls":cls_neg}

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
            classifier_weights = self.class_embeddings[:1+self.num_base,:]   # (num_base+1, dim_emb) == (1+25,dim_emb) (0,1,2,...,25)
        elif split == "novel":
            classifier_weights = torch.cat((
                self.class_embeddings[None,0,:],
                self.class_embeddings[self.num_base+1:,:]
            ),dim=0)
            # self.class_embeddings[self.num_base+1:,:]  # (num_novel, dim_emb) == (1+10,dim_emb)  (0,26,27,...,35)
        elif split == "all":
            classifier_weights = self.class_embeddings                 # (36, dim_emb)   include __background__
        else:
            assert False, "split must be base, novel, or all"
        
        self.classifier_weights = classifier_weights
        self.cls_split = split


    def forward_inference_bsz1(self,input_data):
        ## NOTE this func is is supplemented latter
        '''
        input_data is det_info (Compatible with previous code for TrajCls) or det_info["features"] (for RelationCls)
        
        det_info = {
                    "fstart":torch.as_tensor(fstarts), # shape == (num_traj,)
                    "scores":torch.as_tensor(scores),  # shape == (num_traj,)
                    "bboxes":bboxes,  # list[tensor] , len== num_traj, each shape == (num_boxes, 4)
                    "features":torch.from_numpy(traj_features)  # shape == (num_traj, 2048)
        }
        '''
        if isinstance(input_data,dict):
            det_features = input_data["features"]
        else:
            det_features = input_data
        
        traj_embs = self.V2L_projection(det_features) # (n_det, dim_emb)
        cls_logits = torch.matmul(traj_embs,self.classifier_weights.t()) # (n_det, num_cls)  # include __background__
        cls_probs = torch.softmax(cls_logits,dim=-1)  # (N_det, num_cls) , include __background__
        
        # we filter out background prediction at test time (by set `cls_probs[:,0] = 0.0`),
        # while still keep their forground low score, because we do softmax by considering background)
        cls_probs[:,0] = 0.0

        _,num_cls = cls_probs.shape
        scores,cls_ids = torch.max(cls_probs,dim=-1)  # (n_det,)

        if self.cls_split == "base":
            assert num_cls == self.num_base + 1,"num_cls={},num_base={}".format(num_cls,self.num_base) # 26
            # range of cls_id: 0 ~ 25, (0 is impossible because `cls_probs[:,0] = 0.0`, the same below)
            
        elif self.cls_split == "novel":
            assert num_cls == self.num_novel +  1  # 11
            # range of cls_id: 0 ~ 10, (0 is impossible)
            cls_ids += self.num_base    # range: 0 ~ 10  --> 25 ~ 35 (and 25 is impossible)
        elif self.cls_split == "all":
            assert num_cls == self.num_base + self.num_novel + 1   # 36
            # range of cls_id: 0 ~ 35, (0 is impossible)
        else:
            assert False, "eval_split must be base, novel, or all"
        
        
        return scores,cls_ids


    def forward_inference_bsz1(self,traj_feats_or_embds,input_emb=False):
        if input_emb:  # for teacher model
            traj_embs = traj_feats_or_embds
        else:
            traj_features = traj_feats_or_embds
            traj_embs = self.V2L_projection(traj_features) # (n_det, dim_emb)
        
        cls_logits = torch.matmul(traj_embs,self.classifier_weights.t()) # (n_det, num_cls)  # include __background__
        cls_probs = torch.softmax(cls_logits,dim=-1)  # (N_det, num_cls) , include __background__
        
        # we filter out background prediction at test time (by set `cls_probs[:,0] = 0.0`),
        # while still keep their forground low score, because we do softmax by considering background)
        cls_probs[:,0] = 0.0

        _,num_cls = cls_probs.shape
        scores,cls_ids = torch.max(cls_probs,dim=-1)  # (n_det,)

        if self.cls_split == "base":
            assert num_cls == self.num_base + 1,"num_cls={},num_base={}".format(num_cls,self.num_base) # 26
            # range of cls_id: 0 ~ 25, (0 is impossible because `cls_probs[:,0] = 0.0`, the same below)
            
        elif self.cls_split == "novel":
            assert num_cls == self.num_novel +  1  # 11
            # range of cls_id: 0 ~ 10, (0 is impossible)
            cls_ids += self.num_base    # range: 0 ~ 10  --> 25 ~ 35 (and 25 is impossible)
        elif self.cls_split == "all":
            assert num_cls == self.num_base + self.num_novel + 1   # 36
            # range of cls_id: 0 ~ 35, (0 is impossible)
        else:
            assert False, "eval_split must be base, novel, or all"
        
        
        return scores,cls_ids 


    
    def predict_cls_ids(self,batch_traj_embs,n_dets):
        # NOTE: this function can be called externally, (with inputing traj embedding from the teacher model)
        # and if so, one should first call `self.reset_classifier_weights`
        # batch_traj_embs.shape == (N_det, dim_emb)

        if isinstance(batch_traj_embs,list):
            batch_traj_embs = torch.cat(batch_traj_embs,dim=0)
        
        cls_logits = torch.matmul(batch_traj_embs,self.classifier_weights.t()) # (N_det, num_cls)  # include __background__
        cls_probs = torch.softmax(cls_logits,dim=-1)  # (N_det, num_cls) , include __background__
        
          
        # we filter out background prediction at test time (by set `cls_probs[:,0] = 0.0`),
        # while still keep their forground low score, because we do softmax by considering background)
        cls_probs[:,0] = 0.0

        _,num_cls = cls_probs.shape
        scores,cls_ids = torch.max(cls_probs,dim=-1)  # (N_det,)

        if self.cls_split == "base":
            assert num_cls == self.num_base + 1,"num_cls={},num_base={}".format(num_cls,self.num_base) # 26
            # range of cls_id: 0 ~ 25, (0 is impossible because `cls_probs[:,0] = 0.0`, the same below)
            
        elif self.cls_split == "novel":
            assert num_cls == self.num_novel +  1  # 11
            # range of cls_id: 0 ~ 10, (0 is impossible)
            cls_ids += self.num_base    # range: 0 ~ 10  --> 25 ~ 35 (and 25 is impossible)
        elif self.cls_split == "all":
            assert num_cls == self.num_base + self.num_novel + 1   # 36
            # range of cls_id: 0 ~ 35, (0 is impossible)
        else:
            assert False, "eval_split must be base, novel, or all"
        
        scores = torch.split(scores,n_dets)
        cls_ids = torch.split(cls_ids,n_dets)
        
        return scores,cls_ids



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.vIoU_th = configs["vIoU_th"]
        self.num_base = configs["num_base"]
        self.num_novel = configs["num_novel"]
        self.text_emb_path = configs["text_emb_path"]
        self.loss_factor = configs["loss_factor"]

        text_embeddings = np.load(self.text_emb_path).astype('float32')    # __background__ is all 0's shape == (36,dim_emb)
        self.class_embeddings = nn.Parameter(torch.from_numpy(text_embeddings),requires_grad=False)  # (36,dim_emb)
        
        self.V2L_projection = nn.Sequential(
            nn.Linear(self.dim_roi,self.dim_hid),
            nn.ReLU(),
            nn.Linear(self.dim_hid,self.dim_hid),
            nn.ReLU(),
            nn.Linear(self.dim_hid,self.dim_emb)   # distillation_targets have both positive & negative values, so here we do not end with ReLU
        )
        
        if self.is_train:
            self.reset_classifier_weights("base")
        else:
            self.reset_classifier_weights("novel")
            # This func can be called externally (before call self.forward_inference) to evaluate on other split 
            # (e.g., evaluate on "base" or "all")


    def forward(self,batch_det_features,batch_labels,batch_tgt_features,batch_tgt_traj_ids):
        
        '''
            batch_det_features: list[tensor] each shape == (n_det,2048)
            batch_labels: list[tensor|None]  each item is a tensor of  shape (n_det,) or None
            batch_tgt_features:  list[tensor]  each shape == (n_det_, dim_emb)
            batch_tgt_traj_ids:  list[tensor]  each shape == (n_det_,),  ids w.r.t `n_det`
        '''
        
        if self.use_distillation:
            batch_distil_tgts = (batch_tgt_features,batch_tgt_traj_ids)
        else:
            batch_distil_tgts = None

        n_dets = [x.shape[0] for x in batch_det_features]
        batch_det_features = torch.cat(batch_det_features,dim=0)  # (N_det,2048)
        batch_traj_embs = self.V2L_projection(batch_det_features) # (N_det, dim_emb)

        traj_ids_wrt_batch = torch.as_tensor(range(sum(n_dets)),device=batch_traj_embs.device)  # (N_det,)
        traj_ids_wrt_batch = torch.split(traj_ids_wrt_batch,n_dets)

        traj_ids_for_cls = [ids for bid,ids in enumerate(traj_ids_wrt_batch) if not (batch_labels[bid] is None)]
        
        device_ = batch_traj_embs.device
        if traj_ids_for_cls == []:
            cls_pos = torch.zeros(size=(),device=device_)
            cls_neg = torch.zeros_like(cls_pos)
        else:
            cls_labels = [labels for labels in batch_labels if not (labels is None)]
            traj_ids_for_cls = torch.cat(traj_ids_for_cls,dim=0)  # (N'_det,)
            cls_labels = torch.cat(cls_labels,dim=0)  # (N'_dets,)
            
            traj_embs_for_cls = batch_traj_embs[traj_ids_for_cls,:]  # (N'_dets,dim_emb)
            
            # TODO add sampling func to control the positive-negative ratio
            # sampled_ids = self.pos_neg_sampling(assigned_labels)
            # traj_embs = traj_embs[sampled_ids,:]
            # assigned_labels = assigned_labels[sampled_ids]
            cls_logits = torch.matmul(traj_embs_for_cls,self.classifier_weights.t())  # (N'_dets, dim_emb) x (dim_emb, 1+num_base) --> (N'_dets,1+num_base)
            cls_loss = F.cross_entropy(cls_logits,cls_labels,reduction='none')
            positive_mask = cls_labels > 0
            cls_pos = cls_loss[positive_mask].mean() * self.loss_factor["classification"]
            cls_neg = cls_loss[~positive_mask].mean() * self.loss_factor["classification"]
            if cls_pos.numel() == 0:  
                cls_pos = torch.zeros(size=(),device=device_)
            if cls_neg.numel() == 0:
                cls_neg = torch.zeros(size=(),device=device_)
        
        loss_dict = {"cls_pos":cls_pos,"cls_neg":cls_neg}

        if self.use_distillation:
            tgt_features,tgt_traj_ids = batch_distil_tgts

            traj_ids_for_distil = [ids[tgt_ids] for ids,tgt_ids in zip(traj_ids_wrt_batch,tgt_traj_ids)]
            traj_ids_for_distil = torch.cat(traj_ids_for_distil,dim=0)  # (N''_dets,)

            traj_embs_for_distil = batch_traj_embs[traj_ids_for_distil,:] # (N''_dets, dim_emb)
            tgt_features = torch.cat(tgt_features,dim=0)

            distil_loss = F.l1_loss(traj_embs_for_distil,tgt_features,reduction='mean')
            distil_loss *= self.loss_factor["distillation"]
            loss_dict.update({"distil_loss":distil_loss})

        total_loss = torch.stack(list(loss_dict.values())).sum()    # scalar tensor
        loss_for_show = {k:v.detach() for k,v in loss_dict.items()}
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
        ## approach1: cosine_sim / tau;  refer to OPEN-VOCABULARY OBJECT DETECTION VIA VISION AND LANGUAGE KNOWLEDGE DISTILLATION ICLR2022
        # sim_matrix = cosine_similarity(word_embs,class_embeddings) # (n_det, num_category)
        # sim_matrix = sim_matrix / self.temperature_tau
        
        ## approach2: dot-product, refer to  Open-Vocabulary Object Detection Using Captions CVPR2021
        # NOTE the input of `F.cross_entropy` is logits so we do not nomalize here
        # sim_matrix = torch.matmul(traj_embs,self.classifier_weights.t()) # (n_det, num_category)  <--- we use this


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

        

    def forward_inference(self,batch_det_infos):
        '''
        det_info = batch_det_infos[0]
        det_info = {
                    "fstart":torch.as_tensor(fstarts), # shape == (num_traj,)
                    "scores":torch.as_tensor(scores),  # shape == (num_traj,)
                    "bboxes":bboxes,  # list[tensor] , len== num_traj, each shape == (num_boxes, 4)
                    "features":torch.from_numpy(traj_features)  # shape == (num_traj, 2048)
        }
        '''
        
        batch_det_features = [det_info["features"] for det_info in batch_det_infos]
        n_dets = [x.shape[0] for x in batch_det_features]
        batch_det_features = torch.cat(batch_det_features,dim=0)  # (N_det,2048)
        batch_traj_embs = self.V2L_projection(batch_det_features) # (N_det, dim_emb)
        
        scores,cls_ids = self.predict_cls_ids(batch_traj_embs,n_dets)

        return scores,cls_ids
    

    
    def predict_cls_ids(self,batch_traj_embs,n_dets):
        # NOTE: this function can be called externally, (with inputing traj embedding from the teacher model)
        # and if do the above, one should first call `self.reset_classifier_weights`
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


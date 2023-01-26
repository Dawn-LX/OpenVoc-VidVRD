import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils_func import sigmoid_focal_loss,trajid2pairid

from .PromptModels_v3 import AlproTextEncoder,PromptLearner,PromptLearner_Grouped,FixedPromptEmbdGenerator,PromptLearner_Single
from .PromptModels_v3 import setup_alpro_model  #,get_giou_tags



def _to_predict_cls_ids(
    cls_split,
    num_base,
    num_novel,
    pred_probs,
    pred_topk,
):
    _,num_cls = pred_probs.shape
    scores,cls_ids = torch.topk(pred_probs,pred_topk,dim=-1)  # shape == (n_pair,k)

    # 0,[1,2,....,90,91,92],[93,94,...,130,131,132]

    if cls_split == "base":
        assert num_cls == num_base  # 92
        cls_ids += 1  # 0 ~ 91 --> 1 ~ 92,  len == 92
        
    elif cls_split == "novel":
        assert num_cls == num_novel
        cls_ids += 1 + num_base    # range: 0 ~ 39  --> 93 ~ 132

    elif cls_split == "all":
        assert num_cls == num_base + num_novel  # 132
        cls_ids += 1
        # rang: 0 ~ 131 --> 1 ~ 132
    else:
        assert False, "eval_split must be base, novel, or all"
    
    return scores,cls_ids


class AlproVisual_with_FixedPrompt(nn.Module):
    def __init__(self,configs):
        super().__init__()

        self.num_base = configs["num_base"]
        self.num_novel = configs["num_novel"]
        self.temp_init = configs["temperature_init"]

        ### generate predicate txt embeddings (classifer_weights) via fixed prompt
        pred_cls_split_info_path = configs["pred_cls_split_info_path"]
        alpro_model = setup_alpro_model()
        prompter = FixedPromptEmbdGenerator(alpro_model,pred_cls_split_info_path)
        text_encoder = AlproTextEncoder(alpro_model)

        prompt_type = configs["prompt_type"]
        assert prompt_type == "separate", "Currently, we only support separate prompt for Alpro-visual directly inference"

        if prompt_type == "single":
            prompt_template = configs["prompt_template"]  # e.g., "A video of the visual relation {} between two entities"
            token_embds,token_mask = prompter(prompt_template)  # (132,max_L,768); (132,max_L);
            with torch.no_grad():
                classifier_weights = text_encoder(token_embds,token_mask)  # (132,256)
                
        elif prompt_type == "separate":
            subj_prompt_template = configs["subj_prompt_template"]  # "A video of a person or object {} something"
            obj_prompt_template = configs["obj_prompt_template"]   # "A video of something {} a person or object"

            s_token_embds,s_token_mask = prompter(subj_prompt_template)  # (132,max_L,768); (132,max_L)
            o_token_embds,o_token_mask = prompter(obj_prompt_template)
            with torch.no_grad():
                s_classifier_weights = text_encoder(s_token_embds,s_token_mask)  # (132,256)
                o_classifier_weights = text_encoder(o_token_embds,o_token_mask)

            classifier_weights = torch.cat([
                s_classifier_weights,
                o_classifier_weights
            ],dim=-1) / math.sqrt(2) # (n_cls, 512)
        else:
            print("prompt_type={}, which is not correct".format(prompt_type))
            raise NotImplementedError
        
        self.register_buffer("classifier_weights",classifier_weights,persistent=False)  # (n_cls, 256) or (n_cls, 512)


        self.temperature = self.temp_init

    
    def split_classifier_weights(self,split):

        if split == "base":
            pids_list = list(range(self.num_base))   # (0,1,2,...,91), len==92, 
            # NOTE that 0 index the first base class (not background)
            # because we has exclude __background__ in classifier_weights

        elif split == "novel":
            pids_list = list(range(self.num_base,self.num_base+self.num_novel))
            # (92,94,...,131), len == 40
        elif split == "all":
            pids_list = list(range(self.num_base+self.num_novel))    # i.e., 0 ~ 131, len == 132
        else:
            assert False, "split must be base, novel, or all"
        
        classifier_weights = self.classifier_weights[pids_list,:]
        
        return classifier_weights
    
    def forward_inference_bsz1(self,data,cls_split,pred_topk=10):
        # modified from func:`forward_inference_bsz1` of `AlproPromptTrainer`
        # and remove the MLP of relpos2embd
        # bsz1 means 1 segment
        (
            det_feats,
            traj_embds,
            relpos_feat,
        )   = data


        n_det = traj_embds.shape[0]
        pair_ids = trajid2pairid(n_det).to(traj_embds.device)   # keep the same pair_id order as that in labels
        s_embds = traj_embds[pair_ids[:,0],:]  # (n_pair,256)
        o_embds = traj_embds[pair_ids[:,1],:]  # (n_pair,256)


        s_embds = F.normalize(s_embds,dim=-1)
        o_embds = F.normalize(o_embds,dim=-1)
    
        so_embds = torch.cat([s_embds,o_embds],dim=-1) / math.sqrt(2)  # (n_pair,512)

        classifier_weights = self.split_classifier_weights(cls_split)
        logits = torch.matmul(so_embds,classifier_weights.t()) / self.temperature
     
        pred_probs = torch.sigmoid(logits)  # (n_pair, num_cls) , exclude __background__
        scores,cls_ids = _to_predict_cls_ids(  # (n_pair,k)
            cls_split,
            self.num_base,
            self.num_novel,
            pred_probs,
            pred_topk,
        )
        

        return scores,cls_ids,pair_ids




class OpenVocRelCls_FixedPrompt(nn.Module):
    '''
    unified training, w/o distillation, w/o knowledge from Alpro-visual embedding
    '''
    def __init__(self,configs,is_train=True,train_on_gt_only=False):
        super().__init__()

        self.is_train = is_train
        self.train_on_gt_only = train_on_gt_only
        self.num_base = configs["num_base"]
        self.num_novel = configs["num_novel"]
        self.dim_hid = configs["dim_hidden"]
        self.dim_feat = configs["dim_feat"]
        self.dim_emb = configs["dim_emb"]
        self.temp_init = configs["temperature_init"]

        ### generate predicate txt embeddings (classifer_weights) via fixed prompt
        pred_cls_split_info_path = configs["pred_cls_split_info_path"]
        alpro_model = setup_alpro_model()
        prompter = FixedPromptEmbdGenerator(alpro_model,pred_cls_split_info_path)
        text_encoder = AlproTextEncoder(alpro_model)

        prompt_type = configs["prompt_type"]
        if prompt_type == "single":
            assert self.dim_emb == 256
            prompt_template = configs["prompt_template"]  # e.g., "A video of the visual relation {} between two entities"
            token_embds,token_mask = prompter(prompt_template)  # (132,max_L,768); (132,max_L);
            with torch.no_grad():
                classifier_weights = text_encoder(token_embds,token_mask)  # (132,256)
            
        
        elif prompt_type == "separate":
            assert self.dim_emb == 512
            subj_prompt_template = configs["subj_prompt_template"]  # "A video of a person or object {} something"
            obj_prompt_template = configs["obj_prompt_template"]   # "A video of something {} a person or object"

            s_token_embds,s_token_mask = prompter(subj_prompt_template)  # (132,max_L,768); (132,max_L)
            o_token_embds,o_token_mask = prompter(obj_prompt_template)
            with torch.no_grad():
                s_classifier_weights = text_encoder(s_token_embds,s_token_mask)  # (132,256)
                o_classifier_weights = text_encoder(o_token_embds,o_token_mask)

            classifier_weights = torch.cat([
                s_classifier_weights,
                o_classifier_weights
            ],dim=-1) / math.sqrt(2) # (n_cls, 512)
        else:
            print("prompt_type={}, which is not correct".format(prompt_type))
            raise NotImplementedError
        
        self.register_buffer("classifier_weights",classifier_weights,persistent=False)  # (n_cls, 256) or (n_cls, 512)


        #### Learnable parameters
        self.trajpair_proj = nn.Sequential(
            nn.Linear(self.dim_feat*2,self.dim_hid),
            nn.ReLU(),
            nn.Linear(self.dim_hid,self.dim_hid),
            nn.ReLU(),
            nn.Linear(self.dim_hid,self.dim_emb,bias=False),
        )

        self.relpos2embd = nn.Sequential(
            nn.Linear(12,256),
            nn.ReLU(),
            nn.Linear(256,self.dim_emb,bias=False)
        )

        self.temperature = nn.Parameter(torch.ones([]) * self.temp_init)


    def split_classifier_weights(self,split):

        if split == "base":
            pids_list = list(range(self.num_base))   # (0,1,2,...,91), len==92, 
            # NOTE that 0 index the first base class (not background)
            # because we has exclude __background__ in classifier_weights

        elif split == "novel":
            pids_list = list(range(self.num_base,self.num_base+self.num_novel))
            # (92,94,...,131), len == 40
        elif split == "all":
            pids_list = list(range(self.num_base+self.num_novel))    # i.e., 0 ~ 131, len == 132
        else:
            assert False, "split must be base, novel, or all"
        
        classifier_weights = self.classifier_weights[pids_list,:]
        
        return classifier_weights

    def proj_then_cls(self,so_feats,relpos_feat,cls_split):
        
        so_embds = self.trajpair_proj(so_feats)       # (N_pair,512)
        so_embds = F.normalize(so_embds,dim=-1)       # checked
        
        relpos_embds = self.relpos2embd(relpos_feat)  # (n_pair,512)
        relpos_embds = F.normalize(relpos_embds,dim=-1)

        combined_embds = F.normalize(so_embds+relpos_embds,dim=-1)

        classifier_weights = self.split_classifier_weights(cls_split)
        logits = torch.matmul(combined_embds,classifier_weights.t()) / self.temperature  # (n_pair,num_base)

        return logits


    def forward_on_gt_only(self,batch_data):
        (
            s_roi_feats,      # (bsz,2048)
            o_roi_feats,
            s_embds,          # (bsz,256)
            o_embds,
            relpos_feats,     # (bsz,12)
            triplet_cls_ids   # list[tensor] each shape == (n_preds,3)
        ) = batch_data
        

        bsz = len(triplet_cls_ids)  # bsz == n_pair
        so_feats = torch.cat([s_roi_feats,o_roi_feats],dim=-1)  # (bsz,4096)
        logits = self.proj_then_cls(so_feats,relpos_feats,"base")


        bsz = len(triplet_cls_ids)  # i.e., n_pairs
        multihot = torch.zeros(size=(bsz,self.num_base),device=logits.device)
        for i in range(bsz):
            spo_cls_ids = triplet_cls_ids[i]   # (n_pred,3)
            p_clsids = spo_cls_ids[:,1] - 1  # (n_pred,)  
            # range: 1 ~ num_base  --> 0 ~ num_base -1
            # NOTE this has filtered in dataloader
            multihot[i,p_clsids] = 1

        loss = sigmoid_focal_loss(logits,multihot,reduction='mean')
        loss_for_show = {
            "total(only cls)":loss.detach(),
        }


        return loss,loss_for_show
    
    def forward(self,batch_data,cls_split):
        if self.train_on_gt_only:
            return self.forward_on_gt_only(batch_data)
        (
            batch_det_feats,
            batch_traj_embds,
            batch_relpos_feats,
            batch_labels   
        )   = batch_data
        bsz = len(batch_traj_embds)

        batch_so_feats = []
        for bid in range(bsz):
            traj_embds = batch_traj_embds[bid]
            n_det = traj_embds.shape[0]
            pair_ids = trajid2pairid(n_det).to(traj_embds.device)   # keep the same pair_id order as that in labels
            sids = pair_ids[:,0]
            oids = pair_ids[:,1]
            sf = batch_det_feats[bid][sids]
            of = batch_det_feats[bid][oids]
            so_f = torch.cat([sf,of],dim=-1)  # (n_pair,4096)
            batch_so_feats.append(so_f)
        
        batch_so_feats = torch.cat(batch_so_feats,dim=0) # (N_pair,4096)
        batch_relpos_feats = torch.cat(batch_relpos_feats,dim=0)  # (N_pair,12)
        batch_labels = torch.cat(batch_labels,dim=0)  # shape == (N_pair,133)

        batch_logits = self.proj_then_cls(batch_so_feats,batch_relpos_feats,"base")

        pos_cls,neg_cls = self.cls_loss_on_base(batch_logits,batch_labels)
        total_loss = pos_cls + neg_cls
        loss_for_show = {
            "totoal":total_loss.detach(),
            "pos_cls":pos_cls.detach(),
            "neg_cls":neg_cls.detach(),
        }
        
        return total_loss,loss_for_show


    def cls_loss_on_base(self,logits,labels):
        assert labels.shape[1] == self.num_base + self.num_novel + 1

        pos_mask = torch.any(labels[:,1:self.num_base+1].type(torch.bool),dim=-1)  # (N_pair,)
        # we have filtered out labels whose num_pos == 0 (num_pos w.r.t base_class) refer to `dataset.count_pos_instances`
        # i.e., we can assert pos_mask.sum() > 0
        neg_mask = labels[:,0] > 0  # (N_pair,)

        labels = labels[:,1:self.num_base+1]  # (N_pair,num_base)

        pos_logits = logits[pos_mask,:]
        pos_labels = labels[pos_mask,:]
        pos_loss = sigmoid_focal_loss(pos_logits,pos_labels,reduction='none') #

        neg_logits = logits[neg_mask,:]
        neg_labels = torch.zeros_like(neg_logits)
        neg_loss = sigmoid_focal_loss(neg_logits,neg_labels,reduction='none')
        # perform .mean for a tensor with `.numel()==0` will get `nan`
        if pos_loss.numel() == 0:  
            pos_loss = torch.zeros(size=(),device=labels.device)
        if neg_loss.numel() == 0:
            neg_loss = torch.zeros(size=(),device=labels.device)

        pos_loss = pos_loss.mean()
        neg_loss = neg_loss.mean()

            
        return pos_loss,neg_loss


    def forward_inference_bsz1(self,data,cls_split,pred_topk=10):
        # bsz1 means 1 segment
        (
            det_feats,
            traj_embds,
            relpos_feat,
        )   = data


        n_det = traj_embds.shape[0]
        pair_ids = trajid2pairid(n_det).to(traj_embds.device)   # keep the same pair_id order as that in labels
        sf = det_feats[pair_ids[:,0],:]  # (n_pair,256)
        of = det_feats[pair_ids[:,1],:]  # (n_pair,256)
        so_feats = torch.cat([sf,of],dim=-1) ## (n_pair,4096)
        
        logits = self.proj_then_cls(so_feats,relpos_feat,cls_split)
     
        pred_probs = torch.sigmoid(logits)  # (n_pair, num_cls) , exclude __background__
        scores,cls_ids = _to_predict_cls_ids(  # (n_pair,k)
            cls_split,
            self.num_base,
            self.num_novel,
            pred_probs,
            pred_topk,
        )
        

        return scores,cls_ids,pair_ids



class OpenVocRelCls_LearnablePrompt(nn.Module):
    '''
    unified training  w/ distillation or w/o distillation (if distil_factor < 0)
    '''
    def __init__(self,configs,is_train=True,train_on_gt_only=False):
        super().__init__()

        self.is_train = is_train
        self.train_on_gt_only = train_on_gt_only
        self.num_base = configs["num_base"]
        self.num_novel = configs["num_novel"]
        self.dim_hid = configs["dim_hidden"]
        self.dim_feat = configs["dim_feat"]
        self.dim_emb = configs["dim_emb"]
        self.temp_init = configs["temperature_init"]
        self.distil_factor = configs["distil_factor"]
        n_context = configs["n_context_tokens"]
        pred_cls_split_info_path = configs["pred_cls_split_info_path"]

        alpro_model = setup_alpro_model()
        self.prompter = PromptLearner(n_context,alpro_model,pred_cls_split_info_path,use_pos=False)
        self.text_encoder = AlproTextEncoder(alpro_model)

        self.trajpair_proj = nn.Sequential(
            nn.Linear(self.dim_feat*2,self.dim_hid),
            nn.ReLU(),
            nn.Linear(self.dim_hid,self.dim_hid),
            nn.ReLU(),
            nn.Linear(self.dim_hid,self.dim_emb,bias=False),
        )
        self.relpos2embd = nn.Sequential(
            nn.Linear(12,256),
            nn.ReLU(),
            nn.Linear(256,self.dim_emb,bias=False)
        )

        self.temperature = nn.Parameter(torch.ones([]) * self.temp_init)   


    def state_dict(self):

        state_dict = super().state_dict()

        state_dict_rt = dict()
        for name,val in state_dict.items():
            if name.startswith("text_encoder"):
                continue

            state_dict_rt[name] = val
        
        return state_dict_rt

    
    def load_state_dict(self,state_dict):

        super().load_state_dict(state_dict,strict=False)
    

    def forward_on_gt_only(self,batch_data):
        (
            s_roi_feats,      # (bsz,2048)
            o_roi_feats,
            s_embds,          # (bsz,256)
            o_embds,
            relpos_feats,     # (bsz,12)
            triplet_cls_ids   # list[tensor] each shape == (n_preds,3)
        ) = batch_data
        

        bsz = len(triplet_cls_ids)  # bsz == n_pair

        distil_target = torch.cat([s_embds,o_embds],dim=-1)
        so_feats = torch.cat([s_roi_feats,o_roi_feats],dim=-1)
        so_feats = self.trajpair_proj(so_feats)  # (bsz,512)
        so_feats_norm = F.normalize(so_feats,dim=-1)

        relpos_embds = self.relpos2embd(relpos_feats)  # (n_pair,512)
        relpos_embds = F.normalize(relpos_embds,dim=-1)
        combined_embds = F.normalize(so_feats_norm + relpos_embds,dim=-1)
        

        subj_token_embds,obj_token_embds,token_mask = self.prompter("base")    # (n_cls,max_L,768)
        subj_classifier_weights = self.text_encoder(subj_token_embds,token_mask)  # (n_cls,256)
        obj_classifier_weights = self.text_encoder(obj_token_embds,token_mask)
        
        classifier_weights = torch.cat([
            subj_classifier_weights,
            obj_classifier_weights
        ],dim=-1) / math.sqrt(2) # (n_cls, 512)


        logits = torch.matmul(combined_embds,classifier_weights.t()) / self.temperature  # (bsz,n_cls)


        bsz = len(triplet_cls_ids)  # i.e., n_pairs
        multihot = torch.zeros(size=(bsz,self.num_base),device=logits.device)
        for i in range(bsz):
            spo_cls_ids = triplet_cls_ids[i]   # (n_pred,3)
            p_clsids = spo_cls_ids[:,1] - 1  # (n_pred,)  
            # range: 1 ~ num_base  --> 0 ~ num_base -1
            # NOTE this has filtered in dataloader
            multihot[i,p_clsids] = 1

        cls_loss = sigmoid_focal_loss(logits,multihot,reduction='mean')
        loss_for_show = {
            "cls_loss":cls_loss.detach(),
        }

        if self.distil_factor > 0:
            distil_loss = F.l1_loss(distil_target,so_feats)*self.distil_factor
            loss_for_show.update({"distil_loss":distil_loss.detach()})
            total_loss = cls_loss + distil_loss
        else:
            total_loss = cls_loss

        loss_for_show.update({"total":total_loss.detach()})

        return total_loss,loss_for_show

    
    def forward(self,batch_data,cls_split):
        if self.train_on_gt_only:
            return self.forward_on_gt_only(batch_data)
        (
            batch_det_feats,
            batch_traj_embds,
            batch_relpos_feats,
            batch_labels   
        )   = batch_data
        bsz = len(batch_traj_embds)

        batch_subj_embds = []
        batch_obj_embds = []
        batch_so_feats = []
        for bid in range(bsz):
            traj_embds = batch_traj_embds[bid]
            n_det = traj_embds.shape[0]
            pair_ids = trajid2pairid(n_det).to(traj_embds.device)   # keep the same pair_id order as that in labels
            sids = pair_ids[:,0]
            oids = pair_ids[:,1]
            batch_subj_embds.append(traj_embds[sids,:]) # (n_pair,256)
            batch_obj_embds.append(traj_embds[oids,:])
            sf = batch_det_feats[bid][sids]
            of = batch_det_feats[bid][oids]
            so_f = torch.cat([sf,of],dim=-1)  # (n_pair,4096)
            batch_so_feats.append(so_f)
        
        
        batch_subj_embds = torch.cat(batch_subj_embds,dim=0)  # (N_pair,256)
        batch_obj_embds = torch.cat(batch_obj_embds,dim=0)  # (N_pair,256)
        batch_so_feats = torch.cat(batch_so_feats,dim=0) # (N_pair,4096)
        batch_relpos_feats = torch.cat(batch_relpos_feats,dim=0)
        batch_labels = torch.cat(batch_labels,dim=0)  # shape == (N_pair,133)

        distil_target = torch.cat([batch_subj_embds,batch_obj_embds],dim=-1)  # (N_pair,512)
        
        so_feats = self.trajpair_proj(batch_so_feats)
        so_feats_norm = F.normalize(so_feats,dim=-1)

        relpos_embds = self.relpos2embd(batch_relpos_feats)  # (n_pair,512)
        relpos_embds = F.normalize(relpos_embds,dim=-1)
        combined_embds = F.normalize(so_feats_norm + relpos_embds,dim=-1)
        
        subj_token_embds,obj_token_embds,token_mask = self.prompter("base")    # (num_base,max_L,768)
        subj_classifier_weights = self.text_encoder(subj_token_embds,token_mask)  # (num_base,256)
        obj_classifier_weights = self.text_encoder(obj_token_embds,token_mask)    # has been normalized in text_encoder
        
        classifier_weights = torch.cat([
            subj_classifier_weights,
            obj_classifier_weights
        ],dim=-1) / math.sqrt(2)  # (num_base, 512)
        
        logits = torch.matmul(combined_embds,classifier_weights.t()) / self.temperature
        
        pos_cls,neg_cls = self.cls_loss_on_base(logits,batch_labels)

        loss_for_show = {
            "pos_cls":pos_cls.detach(),
            "neg_cls":neg_cls.detach(),
        }

        if self.distil_factor > 0:
            distil_loss = F.l1_loss(distil_target,so_feats)*self.distil_factor
            loss_for_show.update({"distil_loss":distil_loss.detach()})
            total_loss = pos_cls + neg_cls + distil_loss
        else:
            total_loss = pos_cls + neg_cls

        loss_for_show.update({"total":total_loss.detach()})

        
        return total_loss,loss_for_show


    def cls_loss_on_base(self,logits,labels):
        assert labels.shape[-1] == 1 + self.num_base + self.num_novel

        pos_mask = torch.any(labels[:,1:self.num_base+1].type(torch.bool),dim=-1)  # (N_pair,)
        # we have filtered out labels whose num_pos == 0 (num_pos w.r.t base_class) refer to `dataset.count_pos_instances`
        # i.e., we can assert pos_mask.sum() > 0
        neg_mask = labels[:,0] > 0  # (N_pair,)

        labels = labels[:,1:self.num_base+1]  # (N_pair,num_base)

        pos_logits = logits[pos_mask,:]
        pos_labels = labels[pos_mask,:]
        pos_loss = sigmoid_focal_loss(pos_logits,pos_labels,reduction='none') #

        neg_logits = logits[neg_mask,:]
        neg_labels = torch.zeros_like(neg_logits)
        neg_loss = sigmoid_focal_loss(neg_logits,neg_labels,reduction='none')
        # perform .mean for a tensor with `.numel()==0` will get `nan`
        if pos_loss.numel() == 0:  
            pos_loss = torch.zeros(size=(),device=labels.device)
        if neg_loss.numel() == 0:
            neg_loss = torch.zeros(size=(),device=labels.device)

        pos_loss = pos_loss.mean()
        neg_loss = neg_loss.mean()
            
        return pos_loss,neg_loss

    def reset_classifier_weights(self,cls_split):
        # this is used in test , reset for each epoch

        # for each epoch， reset once and save the classifier_weights as buffer,
        # reset at each iteration is not necessary and is too time consuming
        # and we must re-reset for each epoch

        subj_token_embds,obj_token_embds,token_mask = self.prompter(cls_split)    # (num_base,max_L,768)
        subj_classifier_weights = self.text_encoder(subj_token_embds,token_mask)  # (num_base,256)
        obj_classifier_weights = self.text_encoder(obj_token_embds,token_mask)    # has been normalized in text_encoder
        
        classifier_weights = torch.cat([
            subj_classifier_weights,
            obj_classifier_weights
        ],dim=-1) / math.sqrt(2)  # (num_base, 512)

        self.register_buffer("classifier_weights",classifier_weights,persistent=False)


    def forward_inference_bsz1(self,data,cls_split,pred_topk=10):
        # bsz1 means 1 segment
        (
            det_feats,
            traj_embds,
            rel_pos_feat,
        )   = data


        n_det = det_feats.shape[0]
        pair_ids = trajid2pairid(n_det).to(det_feats.device)   # keep the same pair_id order as that in labels
        s_feats = det_feats[pair_ids[:,0],:]  # (n_pair,256)
        o_feats = det_feats[pair_ids[:,1],:]  # (n_pair,256)
        so_feats = torch.cat([s_feats,o_feats],dim=-1)
        so_feats = self.trajpair_proj(so_feats)
        so_feats_norm = F.normalize(so_feats,dim=-1)

        relpos_embds = self.relpos2embd(rel_pos_feat)  # (n_pair,512)
        relpos_embds = F.normalize(relpos_embds,dim=-1)
        combined_embds = F.normalize(so_feats_norm + relpos_embds,dim=-1)


        logits = torch.matmul(combined_embds,self.classifier_weights.t()) / self.temperature
     
        pred_probs = torch.sigmoid(logits)  # (n_pair, num_cls) , exclude __background__
        scores,cls_ids = _to_predict_cls_ids(  # (n_pair,k)
            cls_split,
            self.num_base,
            self.num_novel,
            pred_probs,
            pred_topk,
        )
        

        return scores,cls_ids,pair_ids


class OpenVocRelCls_stage2(nn.Module):

    def __init__(self,configs,is_train=True,train_on_gt_only=False):
        super().__init__()

        self.is_train = is_train
        self.train_on_gt_only = train_on_gt_only
        self.num_base = configs["num_base"]
        self.num_novel = configs["num_novel"]
        self.dim_hid = configs["dim_hidden"]
        self.dim_feat = configs["dim_feat"]
        self.dim_emb = configs["dim_emb"]
        self.temp_init = configs["temperature_init"]
        self.n_context = configs["n_context_tokens"]
        self.prompter_ckpt_path = configs["prompter_ckpt_path"]
        self.pred_cls_split_info_path = configs["pred_cls_split_info_path"]

        self.setup_model()


    def setup_model(self):

        alpro_model = setup_alpro_model()
        text_encoder = AlproTextEncoder(alpro_model)
        prompter = PromptLearner(self.n_context,alpro_model,self.pred_cls_split_info_path)
        check_point = torch.load(self.prompter_ckpt_path,map_location=torch.device("cpu"))
        state_dict = check_point["model_state_dict"]
        prompter.load_state_dict(state_dict)
        prompter.eval()
        text_encoder.eval()

        with torch.no_grad():
            subj_token_embds,obj_token_embds,token_mask = prompter("all")
            subj_classifier_weights = text_encoder(subj_token_embds,token_mask)  # (132,256)
            obj_classifier_weights = text_encoder(obj_token_embds,token_mask)   # has been normalized in text_encoder


        classifier_weights = torch.cat([
            subj_classifier_weights,
            obj_classifier_weights
        ],dim=-1) / math.sqrt(2)   # (132, 512)

        self.register_buffer("classifier_weights",classifier_weights,persistent=False)  # (132,256)

        self.trajpair_proj = nn.Sequential(
            nn.Linear(self.dim_feat*2,self.dim_hid),
            nn.ReLU(),
            nn.Linear(self.dim_hid,self.dim_hid),
            nn.ReLU(),
            nn.Linear(self.dim_hid,self.dim_emb,bias=False),
        )

        self.temperature = nn.Parameter(torch.ones([]) * self.temp_init)   

        self.relpos2embd = prompter.relpos2embd  # fix this 
        # for p in self.relpos2embd.parameters():
        #     p.requires_grad = False
        
    
    def split_classifier_weights(self,split):

        if split == "base":
            pids_list = list(range(self.num_base))   # (0,1,2,...,91), len==92, 
            # NOTE that 0 index the first base class (not background)
            # because we has exclude __background__ in classifier_weights

        elif split == "novel":
            pids_list = list(range(self.num_base,self.num_base+self.num_novel))
            # (92,94,...,131), len == 40
        elif split == "all":
            pids_list = list(range(self.num_base+self.num_novel))    # i.e., 0 ~ 131, len == 132
        else:
            assert False, "split must be base, novel, or all"
        
        classifier_weights = self.classifier_weights[pids_list,:]
        
        return classifier_weights


    def forward(self,batch_data,cls_split):
        assert not self.train_on_gt_only
        '''
        batch_xxx are lists, in which each item is as following:

        det_feats,              (n_det,2048)
        traj_embds,             (n_pair,256)
        rel_pos_feat,           (n_pair,12)  
        labels                  (n_pair,num_pred_cats)     # num_pred_cats = 1 + num_base + num_novel
        '''
        (
            batch_det_feats,
            batch_traj_embds,
            batch_rel_pos_feat,
            batch_labels
        )   = batch_data
        bsz = len(batch_det_feats)

        batch_subj_feats = []
        batch_obj_feats = []
        for bid in range(bsz):
            traj_feats = batch_det_feats[bid]
            n_det = traj_feats.shape[0]
            pair_ids = trajid2pairid(n_det).to(traj_feats.device)   # keep the same pair_id order as that in labels
            s_feats = traj_feats[pair_ids[:,0],:]  # (n_pair,2048)
            o_feats = traj_feats[pair_ids[:,1],:]  # (n_pair,2048)
            batch_subj_feats.append(s_feats)
            batch_obj_feats.append(o_feats)
        
        batch_subj_feats = torch.cat(batch_subj_feats,dim=0)  # (N_pair,2048)
        batch_obj_feats = torch.cat(batch_obj_feats,dim=0)  # (N_pair,2048)
        batch_rel_pos_feat = torch.cat(batch_rel_pos_feat,dim=0)  # (N_pair,12)
        batch_labels = torch.cat(batch_labels,dim=0)  # shape == (N_pair,num_cls)

        batch_logits = self.proj_then_cls(batch_subj_feats,batch_obj_feats,batch_rel_pos_feat,"base")
        loss = self.loss_on_base(batch_logits,batch_labels)
        return loss


    def proj_then_cls(self,s_feats,o_feats,relpos_feat,cls_split):
        

        so_feats = torch.cat([s_feats,o_feats],dim=-1)  # (N_pair,4096)
        so_embds = self.trajpair_proj(so_feats)       # (N_pair,512)
        so_embds = F.normalize(so_embds,dim=-1)       # checked
        
        relpos_embds = self.relpos2embd(relpos_feat)  # (n_pair,512)
        relpos_embds = F.normalize(relpos_embds,dim=-1)

        combined_embds = F.normalize(so_embds+relpos_embds,dim=-1)

        classifier_weights = self.split_classifier_weights(cls_split)
        logits = torch.matmul(combined_embds,classifier_weights.t()) / self.temperature  # (n_pair,num_base)

        return logits
        

    def loss_on_base(self,logits,labels):

        pos_mask = torch.any(labels[:,1:self.num_base+1].type(torch.bool),dim=-1)  # (N_pair,)
        # we have filtered out labels whose num_pos == 0 (num_pos w.r.t base_class) refer to `dataset.count_pos_instances`
        # i.e., we can assert pos_mask.sum() > 0
        neg_mask = labels[:,0] > 0  # (N_pair,)

        labels = labels[:,1:self.num_base+1]  # (N_pair,num_base)

        pos_logits = logits[pos_mask,:]
        pos_labels = labels[pos_mask,:]
        pos_loss = sigmoid_focal_loss(pos_logits,pos_labels,reduction='none') #

        neg_logits = logits[neg_mask,:]
        neg_labels = torch.zeros_like(neg_logits)
        neg_loss = sigmoid_focal_loss(neg_logits,neg_labels,reduction='none')
        # perform .mean for a tensor with `.numel()==0` will get `nan`
        if pos_loss.numel() == 0:  
            pos_loss = torch.zeros(size=(),device=labels.device)
        if neg_loss.numel() == 0:
            neg_loss = torch.zeros(size=(),device=labels.device)

        pos_loss = pos_loss.mean()
        neg_loss = neg_loss.mean()

        total_loss = pos_loss + neg_loss
        loss_for_show = {
            "total":total_loss.detach(),
            "pos_cls":pos_loss.detach(),
            "neg_cls":neg_loss.detach(),
        }
            
        return total_loss,loss_for_show

    def forward_inference_bsz1(self,data,cls_split,pred_topk=10):
        (
            det_feats,
            traj_embds,
            relpos_feat,
        )   = data


        n_det = det_feats.shape[0]
        pair_ids = trajid2pairid(n_det).to(det_feats.device)   # keep the same pair_id order as that in labels
        s_feats = det_feats[pair_ids[:,0],:]  # (n_pair,2048)
        o_feats = det_feats[pair_ids[:,1],:]  # (n_pair,2048)
        
        logits = self.proj_then_cls(s_feats,o_feats,relpos_feat,cls_split)

        pred_probs = torch.sigmoid(logits)  # (n_pair, num_cls) , exclude __background__
        scores,cls_ids = _to_predict_cls_ids(  # (n_pair,k)
            cls_split,
            self.num_base,
            self.num_novel,
            pred_probs,
            pred_topk,
        )
        
        return scores,cls_ids,pair_ids

class OpenVocRelCls_stage2_Grouped(nn.Module):

    def __init__(self,configs,is_train=True,train_on_gt_only=False):
        super().__init__()

        self.fullysupervise = configs.get("fullysupervise",False)
        self.is_train = is_train
        self.train_on_gt_only = train_on_gt_only
        self.num_base = configs["num_base"]
        self.num_novel = configs["num_novel"]
        self.dim_hid = configs["dim_hidden"]
        self.dim_feat = configs["dim_feat"]
        self.dim_emb = configs["dim_emb"]
        self.temp_init = configs["temperature_init"]
        self.n_context = configs["n_context_tokens"]
        self.n_groups = configs["n_prompt_groups"]
        self.giou_th = configs["giou_th"]
        self.prompter_ckpt_path = configs["prompter_ckpt_path"]
        self.pred_cls_split_info_path = configs["pred_cls_split_info_path"]

        self.setup_model()


    def setup_model(self):

        alpro_model = setup_alpro_model()
        text_encoder = AlproTextEncoder(alpro_model)
        prompter = PromptLearner_Grouped(self.n_groups,self.n_context,alpro_model,self.pred_cls_split_info_path)
        check_point = torch.load(self.prompter_ckpt_path,map_location=torch.device("cpu"))
        state_dict = check_point["model_state_dict"]
        prompter.load_state_dict(state_dict)
        prompter.eval()
        text_encoder.eval()

        with torch.no_grad():
            subj_token_embds,obj_token_embds,token_mask = prompter("all")  # (n_grp*132,max_L,768)
            subj_classifier_weights = text_encoder(subj_token_embds,token_mask)  # (n_grp*132,256)
            obj_classifier_weights = text_encoder(obj_token_embds,token_mask)   # has been normalized in text_encoder


        classifier_weights = torch.cat([
            subj_classifier_weights,
            obj_classifier_weights
        ],dim=-1) / math.sqrt(2)   # (n_grp*132, 512)
        classifier_weights = classifier_weights.reshape(self.n_groups,-1,512)

        self.register_buffer("group_classifier_weights",classifier_weights,persistent=False)  # (n_grp,132, 512)

        self.trajpair_proj = nn.Sequential(
            nn.Linear(self.dim_feat*2,self.dim_hid),
            nn.ReLU(),
            nn.Linear(self.dim_hid,self.dim_hid),
            nn.ReLU(),
            nn.Linear(self.dim_hid,self.dim_emb,bias=False),
        )

        self.temperature = nn.Parameter(torch.ones([]) * self.temp_init)   

        self.relpos2embd = prompter.relpos2embd  # fix this 
        # for p in self.relpos2embd.parameters():
        #     p.requires_grad = False
        
    
    def get_giou_tags(self,rel_gious,giou_th):

        tag_keys = torch.as_tensor(
            [[False, False, False],
            [False, False,  True],
            [False,  True,  True],
            [ True, False, False],
            [ True,  True, False],
            [ True,  True,  True]],device=rel_gious.device
        )
        
        s_tags = rel_gious[:,0] >= giou_th  # (n_pair,)
        e_tags = rel_gious[:,1] >= giou_th
        diff_tags = (rel_gious[:,1] - rel_gious[:,0]) >= 0  # (n_pair,)

        giou_tags = torch.stack([s_tags,e_tags,diff_tags],dim=-1)  # (n_pair,3)
        giou_tags_ = torch.cat([tag_keys,giou_tags],dim=0)  # (6+n_pair,3) 
        #NOTE must manually add `tag_keys` before the unique operation, 
        # because `giou_tags` might not cover all the cases.

        uniq_tags,inverse_ids,count = torch.unique(giou_tags_,return_counts=True,return_inverse=True,dim=0,sorted=True)
        assert len(count) == 6
        inverse_ids = inverse_ids[6:] # exclude the first 6 ids, because they are belonging to `tag_keys`
        count = count - 1
        # uniq_tags = torch.as_tensor(
        #     [[False, False, False],
        #     [False, False,  True],
        #     [False,  True,  True],
        #     [ True, False, False],
        #     [ True,  True, False],
        #     [ True,  True,  True]]
        # )
        # uniq_tags.shape == (6,3)
        # count.shape == (6,)
        # +++,++-,
        # +--,-++,  (i.e., +- can't be +-+; -+ can't be -+-)
        # --+,---

        return giou_tags,inverse_ids,count

    def split_classifier_weights(self,split):

        if split == "base":
            pids_list = list(range(self.num_base))   # (0,1,2,...,91), len==92, 
            # NOTE that 0 index the first base class (not background)
            # because we has exclude __background__ in classifier_weights

        elif split == "novel":
            pids_list = list(range(self.num_base,self.num_base+self.num_novel))
            # (92,94,...,131), len == 40
        elif split == "all":
            pids_list = list(range(self.num_base+self.num_novel))    # i.e., 0 ~ 131, len == 132
        else:
            assert False, "split must be base, novel, or all"
        
        classifier_weights = self.group_classifier_weights[:,pids_list,:] # (n_grp, n_cls, 512)
        
        return classifier_weights


    def forward(self,batch_data,cls_split):
        assert not self.train_on_gt_only
        '''
        batch_xxx are lists, in which each item is as following:

        det_feats,              (n_det,2048)
        traj_embds,             (n_pair,256)
        rel_pos_feat,           (n_pair,12)  
        labels                  (n_pair,num_pred_cats)     # num_pred_cats = 1 + num_base + num_novel
        '''
        (
            batch_det_feats,
            batch_traj_embds,
            batch_rel_pos_giou,
            batch_labels
        )   = batch_data
        bsz = len(batch_det_feats)

        batch_so_feats = []
        batch_relpos = []
        batch_relgiou = []
        for bid in range(bsz):
            traj_feats = batch_det_feats[bid]
            n_det = traj_feats.shape[0]
            pair_ids = trajid2pairid(n_det).to(traj_feats.device)   # keep the same pair_id order as that in labels
            s_feats = traj_feats[pair_ids[:,0],:]  # (n_pair,2048)
            o_feats = traj_feats[pair_ids[:,1],:]  # (n_pair,2048)
            so_feats = torch.cat([s_feats,o_feats],dim=-1)  # (n_pair,4096)
            relpos,giou = batch_rel_pos_giou[bid]

            batch_so_feats.append(so_feats)
            batch_relpos.append(relpos)
            batch_relgiou.append(giou)
        
        batch_so_feats = torch.cat(batch_so_feats,dim=0)  # (N_pair,4096)
        batch_relpos = torch.cat(batch_relpos,dim=0)  # (N_pair,12)
        batch_relgiou = torch.cat(batch_relgiou,dim=0)
        
        batch_labels = torch.cat(batch_labels,dim=0)  # shape == (N_pair,num_cls)
        
        if self.fullysupervise:
            _cls_split = "all"
            loss_func = self.loss_on_all
        else:
            _cls_split = "base"
            loss_func = self.loss_on_base
        
        batch_logits = self.proj_then_cls(batch_so_feats,batch_relpos,batch_relgiou,_cls_split)

        loss = loss_func(batch_logits,batch_labels)
        return loss


    def proj_then_cls(self,so_feats,relpos_feats,rel_gious,cls_split):
        
        so_embds = self.trajpair_proj(so_feats)       # (N_pair,512)
        so_embds = F.normalize(so_embds,dim=-1)       # checked
        relpos_embds = self.relpos2embd(relpos_feats)  # (N_pair,512)
        relpos_embds = F.normalize(relpos_embds,dim=-1)
        combined_embds = F.normalize(so_embds+relpos_embds,dim=-1) # (N_pair,512)
        combined_embds = combined_embds[:,:,None]

        giou_tags,prompt_ids,counts = self.get_giou_tags(rel_gious,self.giou_th)
        classifier_weights = self.split_classifier_weights(cls_split)  # (n_grp,n_cls,512)
        classifier_weights = classifier_weights[prompt_ids,:,:]  # (N_pair,n_cls,512)

        logits = torch.bmm(classifier_weights,combined_embds) / self.temperature 
        # (N_pair,n_cls,512) x (N_pair,512,1) --> (N_pair,n_cls,1)
        logits = logits.squeeze(2)  # (N_pair,n_cls)
        
        return logits
        

    def loss_on_base(self,logits,labels):

        pos_mask = torch.any(labels[:,1:self.num_base+1].type(torch.bool),dim=-1)  # (N_pair,)
        # we have filtered out labels whose num_pos == 0 (num_pos w.r.t base_class) refer to `dataset.count_pos_instances`
        # i.e., we can assert pos_mask.sum() > 0
        neg_mask = labels[:,0] > 0  # (N_pair,)

        labels = labels[:,1:self.num_base+1]  # (N_pair,num_base)

        pos_logits = logits[pos_mask,:]
        pos_labels = labels[pos_mask,:]
        pos_loss = sigmoid_focal_loss(pos_logits,pos_labels,reduction='none') #

        neg_logits = logits[neg_mask,:]
        neg_labels = torch.zeros_like(neg_logits)
        neg_loss = sigmoid_focal_loss(neg_logits,neg_labels,reduction='none')
        # perform .mean for a tensor with `.numel()==0` will get `nan`
        if pos_loss.numel() == 0:  
            pos_loss = torch.zeros(size=(),device=labels.device)
        if neg_loss.numel() == 0:
            neg_loss = torch.zeros(size=(),device=labels.device)

        pos_loss = pos_loss.mean()
        neg_loss = neg_loss.mean()

        total_loss = pos_loss + neg_loss
        loss_for_show = {
            "total":total_loss.detach(),
            "pos_cls":pos_loss.detach(),
            "neg_cls":neg_loss.detach(),
        }
            
        return total_loss,loss_for_show

    def loss_on_all(self,logits,labels):
        pos_mask = torch.any(labels[:,1:].type(torch.bool),dim=-1)  # (N_pair,)
        neg_mask = labels[:,0] > 0  # (N_pair,)

        labels = labels[:,1:]  # (N_pair,num_base+num_novel)

        pos_logits = logits[pos_mask,:]
        pos_labels = labels[pos_mask,:]
        pos_loss = sigmoid_focal_loss(pos_logits,pos_labels,reduction='none') #

        neg_logits = logits[neg_mask,:]
        neg_labels = torch.zeros_like(neg_logits)
        neg_loss = sigmoid_focal_loss(neg_logits,neg_labels,reduction='none')
        # perform .mean for a tensor with `.numel()==0` will get `nan`
        if pos_loss.numel() == 0:  
            pos_loss = torch.zeros(size=(),device=labels.device)
        if neg_loss.numel() == 0:
            neg_loss = torch.zeros(size=(),device=labels.device)

        pos_loss = pos_loss.mean()
        neg_loss = neg_loss.mean()

        total_loss = pos_loss + neg_loss
        loss_for_show = {
            "total":total_loss.detach(),
            "pos_cls":pos_loss.detach(),
            "neg_cls":neg_loss.detach(),
        }
            
        return total_loss,loss_for_show

    def forward_inference_bsz1(self,data,cls_split,pred_topk=10):
        (
            det_feats,
            traj_embds,
            relpos_giou,
        )   = data


        n_det = det_feats.shape[0]
        pair_ids = trajid2pairid(n_det).to(det_feats.device)   # keep the same pair_id order as that in labels
        s_feats = det_feats[pair_ids[:,0],:]  # (n_pair,2048)
        o_feats = det_feats[pair_ids[:,1],:]  # (n_pair,2048)
        so_feats = torch.cat([s_feats,o_feats],dim=-1)
        relpos_feats,rel_gious = relpos_giou

        
        logits = self.proj_then_cls(so_feats,relpos_feats,rel_gious,cls_split)

        pred_probs = torch.sigmoid(logits)  # (n_pair, num_cls) , exclude __background__
        scores,cls_ids = _to_predict_cls_ids(  # (n_pair,k)
            cls_split,
            self.num_base,
            self.num_novel,
            pred_probs,
            pred_topk,
        )
        
        return scores,cls_ids,pair_ids
    
    def forward_inference_bsz1_debug(self,data,cls_split):
        '''
        return the predicate probs of all prompt groups
        '''
        pass
        (
            det_feats,
            traj_embds,
            relpos_giou,
        )   = data


        n_det = det_feats.shape[0]
        pair_ids = trajid2pairid(n_det).to(det_feats.device)   # keep the same pair_id order as that in labels
        s_feats = det_feats[pair_ids[:,0],:]  # (n_pair,2048)
        o_feats = det_feats[pair_ids[:,1],:]  # (n_pair,2048)
        so_feats = torch.cat([s_feats,o_feats],dim=-1)
        relpos_feats,rel_gious = relpos_giou

        
        
        so_embds = self.trajpair_proj(so_feats)       # (N_pair,512)
        so_embds = F.normalize(so_embds,dim=-1)       # checked
        relpos_embds = self.relpos2embd(relpos_feats)  # (N_pair,512)
        relpos_embds = F.normalize(relpos_embds,dim=-1)
        combined_embds = F.normalize(so_embds+relpos_embds,dim=-1) # (N_pair,512)

        giou_tags,prompt_ids,counts = self.get_giou_tags(rel_gious,self.giou_th)
        # prompt_ids.shape == (N_pair,) range: 0~n_grp-1

        classifier_weights = self.split_classifier_weights(cls_split)  # (n_grp,n_cls,512)
        logits_all_grp = []
        for i in range(self.n_groups):
            cw_i = classifier_weights[i,:,:]  # (n_cls,512)
            logits_i = combined_embds @ cw_i.t() / self.temperature  # (N_pair,512) x (512,n_cls) --> (N_pair,n_cls)
            logits_all_grp.append(logits_i)

        logits_all_grp = torch.stack(logits_all_grp,dim=1)  # (N_pair,n_grp,n_cls)

        return prompt_ids,logits_all_grp
        


class OpenVocRelCls_stage2_GroupedRandom(OpenVocRelCls_stage2_Grouped):
    def __init__(self, configs, is_train=True, train_on_gt_only=False):
        super().__init__(configs, is_train, train_on_gt_only)

        ### experiments_RelationCls/_exp_models_v3_TrajBasePredBase/OpenVocRelCls_stage2_Grouped/cfg_ctx10_stage1RandomEP48 is LearnPos2emb
        for p in self.relpos2embd.parameters():
            p.requires_grad = False

    def get_giou_tags(self,rel_gious,giou_th):
    
        n_pair = rel_gious.shape[0]
        random_prompt_ids = torch.randint(0,6,size=(n_pair,),device=rel_gious.device)

        return None,random_prompt_ids,None


class OpenVocRelCls_stage2_MeanEnsemble(OpenVocRelCls_stage2):
    def setup_model(self):
        n_groups = 6
        alpro_model = setup_alpro_model()
        text_encoder = AlproTextEncoder(alpro_model)
        prompter = PromptLearner_Grouped(n_groups,self.n_context,alpro_model,self.pred_cls_split_info_path)
        check_point = torch.load(self.prompter_ckpt_path,map_location=torch.device("cpu"))
        state_dict = check_point["model_state_dict"]
        prompter.load_state_dict(state_dict)
        prompter.eval()
        text_encoder.eval()

        with torch.no_grad():
            subj_token_embds,obj_token_embds,token_mask = prompter("all")  # (n_grp*132,max_L,768)
            n_cls = token_mask.shape[0]//n_groups  # (n_grp*n_cls,max_L)

            subj_token_embds = subj_token_embds.reshape(n_groups,n_cls,-1,768).mean(dim=0)
            obj_token_embds = obj_token_embds.reshape(n_groups,n_cls,-1,768).mean(dim=0)
            token_mask = token_mask[:n_cls,:]   # (n_cls,max_L)

            subj_classifier_weights = text_encoder(subj_token_embds,token_mask)  # (132,256)
            obj_classifier_weights = text_encoder(obj_token_embds,token_mask)   # has been normalized in text_encoder


        classifier_weights = torch.cat([
            subj_classifier_weights,
            obj_classifier_weights
        ],dim=-1) / math.sqrt(2)   # (132, 512)

        self.register_buffer("classifier_weights",classifier_weights,persistent=False)  # (132,256)


        self.trajpair_proj = nn.Sequential(
            nn.Linear(self.dim_feat*2,self.dim_hid),
            nn.ReLU(),
            nn.Linear(self.dim_hid,self.dim_hid),
            nn.ReLU(),
            nn.Linear(self.dim_hid,self.dim_emb,bias=False),
        )

        self.temperature = nn.Parameter(torch.ones([]) * self.temp_init)   

        self.relpos2embd = prompter.relpos2embd  # fix this 
        # for p in self.relpos2embd.parameters():
        #     p.requires_grad = False




class OpenVocRelCls_stage2_Single(nn.Module):

    def __init__(self,configs,is_train=True,train_on_gt_only=False):
        super().__init__()

        self.is_train = is_train
        self.train_on_gt_only = train_on_gt_only
        self.num_base = configs["num_base"]
        self.num_novel = configs["num_novel"]
        self.dim_hid = configs["dim_hidden"]
        self.dim_feat = configs["dim_feat"]
        self.dim_emb = configs["dim_emb"]
        self.temp_init = configs["temperature_init"]
        self.n_context = configs["n_context_tokens"]
        self.prompter_ckpt_path = configs["prompter_ckpt_path"]
        self.pred_cls_split_info_path = configs["pred_cls_split_info_path"]

        self.setup_model()


    def setup_model(self):

        alpro_model = setup_alpro_model()
        text_encoder = AlproTextEncoder(alpro_model)
        prompter = PromptLearner_Single(self.n_context,alpro_model,self.pred_cls_split_info_path)
        check_point = torch.load(self.prompter_ckpt_path,map_location=torch.device("cpu"))
        state_dict = check_point["model_state_dict"]
        prompter.load_state_dict(state_dict)
        prompter.eval()
        text_encoder.eval()

        with torch.no_grad():
            token_embds,token_mask = prompter("all")
            classifier_weights = text_encoder(token_embds,token_mask)  # (132,256) # has been normalized in text_encoder
        
        self.register_buffer("classifier_weights",classifier_weights,persistent=False)  # (132,256)

        self.trajpair_proj = nn.Sequential(
            nn.Linear(self.dim_feat*2,self.dim_hid),
            nn.ReLU(),
            nn.Linear(self.dim_hid,self.dim_hid),
            nn.ReLU(),
            nn.Linear(self.dim_hid,self.dim_emb,bias=False),
        )

        self.temperature = nn.Parameter(torch.ones([]) * self.temp_init)   

        self.relpos2embd = prompter.relpos2embd  # fix this 
        for p in self.relpos2embd.parameters():
            p.requires_grad = False
        
    
    def split_classifier_weights(self,split):

        if split == "base":
            pids_list = list(range(self.num_base))   # (0,1,2,...,91), len==92, 
            # NOTE that 0 index the first base class (not background)
            # because we has exclude __background__ in classifier_weights

        elif split == "novel":
            pids_list = list(range(self.num_base,self.num_base+self.num_novel))
            # (92,94,...,131), len == 40
        elif split == "all":
            pids_list = list(range(self.num_base+self.num_novel))    # i.e., 0 ~ 131, len == 132
        else:
            assert False, "split must be base, novel, or all"
        
        classifier_weights = self.classifier_weights[pids_list,:]
        
        return classifier_weights


    def forward(self,batch_data,cls_split):
        assert not self.train_on_gt_only
        '''
        batch_xxx are lists, in which each item is as following:

        det_feats,              (n_det,2048)
        traj_embds,             (n_pair,256)
        rel_pos_feat,           (n_pair,12)  
        labels                  (n_pair,num_pred_cats)     # num_pred_cats = 1 + num_base + num_novel
        '''
        (
            batch_det_feats,
            batch_traj_embds,
            batch_rel_pos_feat,
            batch_labels
        )   = batch_data
        bsz = len(batch_det_feats)

        batch_subj_feats = []
        batch_obj_feats = []
        for bid in range(bsz):
            traj_feats = batch_det_feats[bid]
            n_det = traj_feats.shape[0]
            pair_ids = trajid2pairid(n_det).to(traj_feats.device)   # keep the same pair_id order as that in labels
            s_feats = traj_feats[pair_ids[:,0],:]  # (n_pair,2048)
            o_feats = traj_feats[pair_ids[:,1],:]  # (n_pair,2048)
            batch_subj_feats.append(s_feats)
            batch_obj_feats.append(o_feats)
        
        batch_subj_feats = torch.cat(batch_subj_feats,dim=0)  # (N_pair,2048)
        batch_obj_feats = torch.cat(batch_obj_feats,dim=0)  # (N_pair,2048)
        batch_rel_pos_feat = torch.cat(batch_rel_pos_feat,dim=0)  # (N_pair,12)
        batch_labels = torch.cat(batch_labels,dim=0)  # shape == (N_pair,num_cls)

        batch_logits = self.proj_then_cls(batch_subj_feats,batch_obj_feats,batch_rel_pos_feat,"base")
        loss = self.loss_on_base(batch_logits,batch_labels)
        return loss


    def proj_then_cls(self,s_feats,o_feats,relpos_feat,cls_split):
        

        so_feats = torch.cat([s_feats,o_feats],dim=-1)  # (N_pair,4096)
        so_embds = self.trajpair_proj(so_feats)       # (N_pair,512)
        so_embds = F.normalize(so_embds,dim=-1)       # checked
        
        relpos_embds = self.relpos2embd(relpos_feat)  # (n_pair,512)
        relpos_embds = F.normalize(relpos_embds,dim=-1)

        combined_embds = F.normalize(so_embds+relpos_embds,dim=-1)

        classifier_weights = self.split_classifier_weights(cls_split)
        logits = torch.matmul(combined_embds,classifier_weights.t()) / self.temperature  # (n_pair,num_base)

        return logits
        

    def loss_on_base(self,logits,labels):

        pos_mask = torch.any(labels[:,1:self.num_base+1].type(torch.bool),dim=-1)  # (N_pair,)
        # we have filtered out labels whose num_pos == 0 (num_pos w.r.t base_class) refer to `dataset.count_pos_instances`
        # i.e., we can assert pos_mask.sum() > 0
        neg_mask = labels[:,0] > 0  # (N_pair,)

        labels = labels[:,1:self.num_base+1]  # (N_pair,num_base)

        pos_logits = logits[pos_mask,:]
        pos_labels = labels[pos_mask,:]
        pos_loss = sigmoid_focal_loss(pos_logits,pos_labels,reduction='none') #

        neg_logits = logits[neg_mask,:]
        neg_labels = torch.zeros_like(neg_logits)
        neg_loss = sigmoid_focal_loss(neg_logits,neg_labels,reduction='none')
        # perform .mean for a tensor with `.numel()==0` will get `nan`
        if pos_loss.numel() == 0:  
            pos_loss = torch.zeros(size=(),device=labels.device)
        if neg_loss.numel() == 0:
            neg_loss = torch.zeros(size=(),device=labels.device)

        pos_loss = pos_loss.mean()
        neg_loss = neg_loss.mean()

        total_loss = pos_loss + neg_loss
        loss_for_show = {
            "total":total_loss.detach(),
            "pos_cls":pos_loss.detach(),
            "neg_cls":neg_loss.detach(),
        }
            
        return total_loss,loss_for_show

    def forward_inference_bsz1(self,data,cls_split,pred_topk=10):
        (
            det_feats,
            traj_embds,
            relpos_feat,
        )   = data


        n_det = det_feats.shape[0]
        pair_ids = trajid2pairid(n_det).to(det_feats.device)   # keep the same pair_id order as that in labels
        s_feats = det_feats[pair_ids[:,0],:]  # (n_pair,2048)
        o_feats = det_feats[pair_ids[:,1],:]  # (n_pair,2048)
        
        logits = self.proj_then_cls(s_feats,o_feats,relpos_feat,cls_split)

        pred_probs = torch.sigmoid(logits)  # (n_pair, num_cls) , exclude __background__
        scores,cls_ids = _to_predict_cls_ids(  # (n_pair,k)
            cls_split,
            self.num_base,
            self.num_novel,
            pred_probs,
            pred_topk,
        )
        
        return scores,cls_ids,pair_ids

class OpenVocRelCls_stage2_Single_Grouped(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        '''
        w/o compositional & w/ motion
        '''


class VidVRDII_FixedPrompt(nn.Module):
    '''
    unified training, w/o distillation, w/o knowledge from Alpro-visual embedding
    '''
    def __init__(self,configs,is_train=True,train_on_gt_only=False):
        super().__init__()

        self.is_train = is_train
        self.train_on_gt_only = train_on_gt_only
        self.num_base = configs["num_base"]
        self.num_novel = configs["num_novel"]
        self.dim_hid = configs["dim_hidden"]
        self.dim_feat = configs["dim_feat"]
        self.dim_emb = configs["dim_emb"]
        self.temp_init = configs["temperature_init"]

        ### generate predicate txt embeddings (classifer_weights) via fixed prompt
        pred_cls_split_info_path = configs["pred_cls_split_info_path"]
        alpro_model = setup_alpro_model()
        prompter = FixedPromptEmbdGenerator(alpro_model,pred_cls_split_info_path)
        text_encoder = AlproTextEncoder(alpro_model)

        prompt_type = configs["prompt_type"]
        if prompt_type == "single":
            assert self.dim_emb == 256
            prompt_template = configs["prompt_template"]  # e.g., "A video of the visual relation {} between two entities"
            token_embds,token_mask = prompter(prompt_template)  # (132,max_L,768); (132,max_L);
            with torch.no_grad():
                classifier_weights = text_encoder(token_embds,token_mask)  # (132,256)
            
        
        elif prompt_type == "separate":
            assert self.dim_emb == 512
            subj_prompt_template = configs["subj_prompt_template"]  # "A video of a person or object {} something"
            obj_prompt_template = configs["obj_prompt_template"]   # "A video of something {} a person or object"

            s_token_embds,s_token_mask = prompter(subj_prompt_template)  # (132,max_L,768); (132,max_L)
            o_token_embds,o_token_mask = prompter(obj_prompt_template)
            with torch.no_grad():
                s_classifier_weights = text_encoder(s_token_embds,s_token_mask)  # (132,256)
                o_classifier_weights = text_encoder(o_token_embds,o_token_mask)

            classifier_weights = torch.cat([
                s_classifier_weights,
                o_classifier_weights
            ],dim=-1) / math.sqrt(2) # (n_cls, 512)
        else:
            print("prompt_type={}, which is not correct".format(prompt_type))
            raise NotImplementedError
        
        self.register_buffer("classifier_weights",classifier_weights,persistent=False)  # (n_cls, 256) or (n_cls, 512)


        #### Learnable parameters
        self.trajpair_proj = nn.Sequential(
            nn.Linear(self.dim_feat*2,self.dim_hid),
            nn.Dropout(0.2),
            nn.ReLU(),
        )

        self.combined2embd = nn.Sequential(
            nn.Linear(12+self.dim_hid,256),
            nn.ReLU(),
        )

        self.temperature = nn.Parameter(torch.ones([]) * self.temp_init)


    def split_classifier_weights(self,split):

        if split == "base":
            pids_list = list(range(self.num_base))   # (0,1,2,...,91), len==92, 
            # NOTE that 0 index the first base class (not background)
            # because we has exclude __background__ in classifier_weights

        elif split == "novel":
            pids_list = list(range(self.num_base,self.num_base+self.num_novel))
            # (92,94,...,131), len == 40
        elif split == "all":
            pids_list = list(range(self.num_base+self.num_novel))    # i.e., 0 ~ 131, len == 132
        else:
            assert False, "split must be base, novel, or all"
        
        classifier_weights = self.classifier_weights[pids_list,:]
        
        return classifier_weights

    def proj_then_cls(self,so_feats,relpos_feat,cls_split):
        
        so_feats = self.trajpair_proj(so_feats)       # (N_pair,512)
        combined_feats = torch.cat([so_feats,relpos_feat],dim=-1)
        combined_embds = self.combined2embd(combined_feats)
        combined_embds = F.normalize(combined_embds,dim=-1)       
        

        classifier_weights = self.split_classifier_weights(cls_split)
        logits = torch.matmul(combined_embds,classifier_weights.t()) / self.temperature  # (n_pair,num_base)

        return logits


    def forward_on_gt_only(self,batch_data):
        (
            s_roi_feats,      # (bsz,2048)
            o_roi_feats,
            s_embds,          # (bsz,256)
            o_embds,
            relpos_feats,     # (bsz,12)
            triplet_cls_ids   # list[tensor] each shape == (n_preds,3)
        ) = batch_data
        

        bsz = len(triplet_cls_ids)  # bsz == n_pair
        so_feats = torch.cat([s_roi_feats,o_roi_feats],dim=-1)  # (bsz,4096)
        logits = self.proj_then_cls(so_feats,relpos_feats,"base")


        bsz = len(triplet_cls_ids)  # i.e., n_pairs
        multihot = torch.zeros(size=(bsz,self.num_base),device=logits.device)
        for i in range(bsz):
            spo_cls_ids = triplet_cls_ids[i]   # (n_pred,3)
            p_clsids = spo_cls_ids[:,1] - 1  # (n_pred,)  
            # range: 1 ~ num_base  --> 0 ~ num_base -1
            # NOTE this has filtered in dataloader
            multihot[i,p_clsids] = 1

        loss = sigmoid_focal_loss(logits,multihot,reduction='mean')
        loss_for_show = {
            "total(only cls)":loss.detach(),
        }


        return loss,loss_for_show
    
    def forward(self,batch_data,cls_split):
        if self.train_on_gt_only:
            return self.forward_on_gt_only(batch_data)
        (
            batch_det_feats,
            batch_traj_embds,
            batch_relpos_feats,
            batch_labels   
        )   = batch_data
        bsz = len(batch_traj_embds)

        batch_so_feats = []
        for bid in range(bsz):
            traj_embds = batch_traj_embds[bid]
            n_det = traj_embds.shape[0]
            pair_ids = trajid2pairid(n_det).to(traj_embds.device)   # keep the same pair_id order as that in labels
            sids = pair_ids[:,0]
            oids = pair_ids[:,1]
            sf = batch_det_feats[bid][sids]
            of = batch_det_feats[bid][oids]
            so_f = torch.cat([sf,of],dim=-1)  # (n_pair,4096)
            batch_so_feats.append(so_f)
        
        batch_so_feats = torch.cat(batch_so_feats,dim=0) # (N_pair,4096)
        batch_relpos_feats = torch.cat(batch_relpos_feats,dim=0)  # (N_pair,12)
        batch_labels = torch.cat(batch_labels,dim=0)  # shape == (N_pair,133)

        batch_logits = self.proj_then_cls(batch_so_feats,batch_relpos_feats,"base")

        pos_cls,neg_cls = self.cls_loss_on_base(batch_logits,batch_labels)
        total_loss = pos_cls + neg_cls
        loss_for_show = {
            "totoal":total_loss.detach(),
            "pos_cls":pos_cls.detach(),
            "neg_cls":neg_cls.detach(),
        }
        
        return total_loss,loss_for_show


    def cls_loss_on_base(self,logits,labels):
        assert labels.shape[1] == self.num_base + self.num_novel + 1

        pos_mask = torch.any(labels[:,1:self.num_base+1].type(torch.bool),dim=-1)  # (N_pair,)
        # we have filtered out labels whose num_pos == 0 (num_pos w.r.t base_class) refer to `dataset.count_pos_instances`
        # i.e., we can assert pos_mask.sum() > 0
        neg_mask = labels[:,0] > 0  # (N_pair,)

        labels = labels[:,1:self.num_base+1]  # (N_pair,num_base)

        pos_logits = logits[pos_mask,:]
        pos_labels = labels[pos_mask,:]
        pos_loss = sigmoid_focal_loss(pos_logits,pos_labels,reduction='none') #

        neg_logits = logits[neg_mask,:]
        neg_labels = torch.zeros_like(neg_logits)
        neg_loss = sigmoid_focal_loss(neg_logits,neg_labels,reduction='none')
        # perform .mean for a tensor with `.numel()==0` will get `nan`
        if pos_loss.numel() == 0:  
            pos_loss = torch.zeros(size=(),device=labels.device)
        if neg_loss.numel() == 0:
            neg_loss = torch.zeros(size=(),device=labels.device)

        pos_loss = pos_loss.mean()
        neg_loss = neg_loss.mean()

            
        return pos_loss,neg_loss


    def forward_inference_bsz1(self,data,cls_split,pred_topk=10):
        # bsz1 means 1 segment
        (
            det_feats,
            traj_embds,
            relpos_feat,
        )   = data


        n_det = traj_embds.shape[0]
        pair_ids = trajid2pairid(n_det).to(traj_embds.device)   # keep the same pair_id order as that in labels
        sf = det_feats[pair_ids[:,0],:]  # (n_pair,256)
        of = det_feats[pair_ids[:,1],:]  # (n_pair,256)
        so_feats = torch.cat([sf,of],dim=-1) ## (n_pair,4096)
        
        logits = self.proj_then_cls(so_feats,relpos_feat,cls_split)
     
        pred_probs = torch.sigmoid(logits)  # (n_pair, num_cls) , exclude __background__
        scores,cls_ids = _to_predict_cls_ids(  # (n_pair,k)
            cls_split,
            self.num_base,
            self.num_novel,
            pred_probs,
            pred_topk,
        )
        

        return scores,cls_ids,pair_ids


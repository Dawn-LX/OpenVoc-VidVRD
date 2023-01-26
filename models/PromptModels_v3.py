# import root_path
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertConfig,BertTokenizerFast
from utils.utils_func import load_json,trajid2pairid,sigmoid_focal_loss,unique_with_idx_nd
from Alpro_modeling.timesformer.helpers import resize_spatial_embedding, resize_temporal_embedding
from Alpro_modeling.alpro_models import AlproBaseModel
from Alpro_config_release.default_alpro_configs import video_retrieval_configs as GeneralAlproCfg


def load_state_dict_with_pos_embed_resizing(model, loaded_state_dict_or_path, 
                                                    num_patches, num_frames, 
                                                    spatial_embed_key='visual_encoder.model.pos_embed', 
                                                    temporal_embed_key='visual_encoder.model.time_embed',
                                                    strict=False,
                                                    remove_text_encoder_prefix=False,
                                                    logger = None,
                                                    ):
    """operated in-place, no need to return `model`,
    
    Used to load e2e model checkpoints.

    remove_text_encoder_prefix: set to True, when finetune downstream models from pre-trained checkpoints.
    """
    if logger is None:
        print_func = print
    else:
        print_func = logger.info

    if isinstance(loaded_state_dict_or_path, str):
        loaded_state_dict = torch.load(
            loaded_state_dict_or_path, map_location="cpu")
        
    else:
        loaded_state_dict = loaded_state_dict_or_path

    new_state_dict = loaded_state_dict.copy()

    for key in loaded_state_dict:
        if 'text_encoder.bert' in key and remove_text_encoder_prefix:
            new_key = key.replace('text_encoder.bert','text_encoder')
            new_state_dict[new_key] = new_state_dict.pop(key)

    loaded_state_dict = new_state_dict

    ## Resizing spatial embeddings in case they don't match
    if num_patches + 1 != loaded_state_dict[spatial_embed_key].size(1):
        loaded_state_dict[spatial_embed_key] = resize_spatial_embedding(loaded_state_dict, spatial_embed_key, num_patches)
    else:
        print_func('The length of spatial position embedding matches. No need to resize.')

    ## Resizing time embeddings in case they don't match
    if temporal_embed_key in loaded_state_dict and num_frames != loaded_state_dict[temporal_embed_key].size(1):
        loaded_state_dict[temporal_embed_key] = resize_temporal_embedding(loaded_state_dict, temporal_embed_key, num_frames)
    else:
        print_func('No temporal encoding found. Or the length of temporal position embedding matches. No need to resize.')

    model_keys = set([k for k in list(model.state_dict().keys())])
    load_keys = set(loaded_state_dict.keys())

    toload = {}
    mismatched_shape_keys = []
    for k in model_keys:
        if k in load_keys:
            if model.state_dict()[k].shape != loaded_state_dict[k].shape:
                mismatched_shape_keys.append(k)
            else:
                toload[k] = loaded_state_dict[k]

    print_func("You can ignore the keys with `num_batches_tracked` or from task heads")
    print_func("Keys in loaded but not in model:")
    diff_keys = load_keys.difference(model_keys)
    # print_func(f"In total {len(diff_keys)}, {sorted(diff_keys)}")
    print_func(f"In total {len(diff_keys)}, ")
    print_func("Keys in model but not in loaded:")
    diff_keys = model_keys.difference(load_keys)
    # print_func(f"In total {len(diff_keys)}, {sorted(diff_keys)}")
    print_func(f"In total {len(diff_keys)}, ")
    print_func("Keys in model and loaded, but shape mismatched:")
    # print_func(f"In total {len(mismatched_shape_keys)}, {sorted(mismatched_shape_keys)}")
    print_func(f"In total {len(mismatched_shape_keys)}, ")
    model.load_state_dict(toload, strict=strict)

def setup_alpro_model():
    print("Setup Alpro-text ...")
    # has to be a BertConfig instance
    model_cfg = load_json(GeneralAlproCfg.model_config)
    model_cfg = BertConfig(**model_cfg)
    # add downstream model config
    add_attr_list = []
    for k in add_attr_list:
        setattr(model_cfg, k, GeneralAlproCfg[k])

    # we separate the CNN and the transformer in order to use different optimizer for each
    # transformer still has a CNN layer inside, used to down sample grid.
    print("setup e2e model")

    video_enc_cfg = load_json(GeneralAlproCfg.visual_model_cfg)
    video_enc_cfg['num_frm'] = GeneralAlproCfg.num_frm
    video_enc_cfg['img_size'] = GeneralAlproCfg.crop_img_size


    model = AlproBaseModel(
        model_cfg,
        input_format=GeneralAlproCfg.img_input_format,
        video_enc_cfg=video_enc_cfg
    )

    

    if GeneralAlproCfg.e2e_weights_path:  # we use this
        print(f"Loading e2e weights from {GeneralAlproCfg.e2e_weights_path}")
        num_patches = (GeneralAlproCfg.crop_img_size // video_enc_cfg['patch_size']) ** 2
        # NOTE strict if False if loaded from ALBEF ckpt
        load_state_dict_with_pos_embed_resizing(model, 
                                                GeneralAlproCfg.e2e_weights_path, 
                                                num_patches=num_patches, 
                                                num_frames=GeneralAlproCfg.num_frm, 
                                                strict=False,
                                                )
    else:
        print(f"Loading visual weights from {GeneralAlproCfg.visual_weights_path}")
        print(f"Loading bert weights from {GeneralAlproCfg.bert_weights_path}")
        model.load_separate_ckpt(
            visual_weights_path=GeneralAlproCfg.visual_weights_path,
            bert_weights_path=GeneralAlproCfg.bert_weights_path
        )

    print("Setup Alpro-text done!")
    return model



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
        assert num_cls == num_base
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


class AlproTextEncoder(nn.Module):
    def __init__(self,alpro_model):
        super().__init__()

        self.max_txt_len = GeneralAlproCfg.max_txt_len
        self.bert = alpro_model.text_encoder.bert
        self.text_proj = alpro_model.text_proj

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self,token_embds,token_mask,output_with_norm=True):
        '''We only implement forward for inference'''
        # token_embds.shape == (bsz, max_L, dim_emb) == (n_str+2, 768), 2 stands for [CLS] & [SEP] tokens
        # NOTE: note that the padded embds are not all-zero, we use the learned embds (indexed by 0) in Alpro-text's bert
        # token_mask.sahpe == (bsz,max_L)

        bsz,max_L,_ = token_embds.shape
        assert max_L == self.max_txt_len

        text_output = self.bert(inputs_embeds=token_embds,
                                             attention_mask=token_mask,
                                             return_dict=True,
                                             mode='text'
                                            )
        text_embeds = text_output.last_hidden_state
        text_feat = self.text_proj(text_embeds[:,0,:])
        if output_with_norm:
            text_feat = F.normalize(text_feat,dim=-1)                 

        return text_feat


class FixedPromptEmbdGenerator(object):

    def __init__(
        self,alpro_model,cls_split_info_path
    ):
        super().__init__()

        cls_split_info = load_json(cls_split_info_path)

        cls2id_map = cls_split_info["cls2id"]
        cls_names = sorted(cls2id_map.items(),key= lambda x:x[1])
        cls_names = [x[0] for x in cls_names]  # including __background__
        self.cls_names = [name.replace("_", " ") for name in cls_names]
        self.name_lens = [len(name.split(" ")) for name in cls_names]
        self.n_cls = len(self.cls_names)
        self.max_txt_len = GeneralAlproCfg.max_txt_len
        self.alpro_model = alpro_model
        
    def __call__(self,prompt_template):
        '''
        e.g.,  prompt_template = "A video of the visual relation {} between two entities"
        prompt_template = "A video of a person or object {} something"  # for separete prompt
        prompt_template = "A video of something {} a person or object"
        '''

        token_strs = [prompt_template.format(name) for name in self.cls_names] # including __background__
        assert all([len(x.split(" ")) <= self.max_txt_len for x in token_strs])

        tokenizer = BertTokenizerFast.from_pretrained("/home/gkf/project/ALPRO/ext/bert-base-uncased")
        batch_enc = tokenizer.batch_encode_plus(
            token_strs,  # bsz= n_cls
            max_length= self.max_txt_len,  # default: 40
            padding='max_length',
            return_tensors="pt",
            truncation=True
        )
        token_ids = batch_enc.input_ids  # (n_cls, max_L)  # max_L == 40
        token_mask = batch_enc.attention_mask  # (n_cls, max_L)
        # [CLS] t1 t2 ... tn [SEP] 0, 0, 0 ..., 0 (zero-padding to max_L == 40) refer to `tools/token_ids.png`

        with torch.no_grad():
            token_embds =  self.alpro_model.text_encoder.bert.embeddings.word_embeddings(token_ids)  # (n_cls,40,768), n_cls  including __background__
        
        token_embds = token_embds[1:,:,:]   # exclude background
        token_mask = token_mask[1:,:]       # exclude background
        
        return token_embds,token_mask


class PromptLearner(nn.Module):

    def __init__(
        self,n_context,alpro_model,cls_split_info_path,use_pos=True
    ):
        super().__init__()

        self.use_pos = use_pos
        cls_split_info = load_json(cls_split_info_path)
        self.num_base = sum([v=="base" for v in cls_split_info["cls2split"].values()])
        self.num_novel = sum([v=="novel" for v in cls_split_info["cls2split"].values()])

        cls2id_map = cls_split_info["cls2id"]
        cls_names = sorted(cls2id_map.items(),key= lambda x:x[1])
        cls_names = [x[0] for x in cls_names]  # including __background__
        cls_names = [name.replace("_", " ") for name in cls_names]
        name_lens = [len(name.split(" ")) for name in cls_names]
        self.n_cls = len(cls_names)
        self.n_ctx = n_context
        self.max_txt_len = GeneralAlproCfg.max_txt_len
        assert all([len_ + self.n_ctx <= self.max_txt_len for len_ in name_lens])

        place_holder_strs = " ".join(["X"] * self.n_ctx)
        token_strs = [place_holder_strs + " " + name for name in cls_names] # including __background__

        tokenizer = BertTokenizerFast.from_pretrained("/home/gkf/project/ALPRO/ext/bert-base-uncased")
        batch_enc = tokenizer.batch_encode_plus(
            token_strs,  # bsz= n_cls
            max_length= self.max_txt_len,  # default: 40
            padding='max_length',
            return_tensors="pt",
            truncation=True
        )
        token_ids = batch_enc.input_ids  # (n_cls, max_L)  # max_L == 40
        token_mask = batch_enc.attention_mask  # (n_cls, max_L)
        # [CLS] t1 t2 ... tn [SEP] 0, 0, 0 ..., 0 (zero-padding to max_L == 40) refer to `tools/token_ids.png`

        with torch.no_grad():
            token_embds =  alpro_model.text_encoder.bert.embeddings.word_embeddings(token_ids)  # (n_cls,40,768), n_cls  including __background__
        
        prefix_embds =  token_embds[:, :1, :]  # (n_cls,1,768), [CLS] token embedding
        suffix_embds =  token_embds[:, 1 + self.n_ctx :, :]  # (n_cls, 40-1-n_ctx ,768) embedding of cls_name tokens, [SEP] token, and zero-pad tokens

        self.register_buffer("prefix_embds", prefix_embds)
        self.register_buffer("suffix_embds", suffix_embds)
        self.register_buffer("token_mask",token_mask)

        self.setup_learnable_parameters()

    def setup_learnable_parameters(self):
        subj_ctx_embds = torch.empty(self.n_ctx, 768)
        nn.init.normal_(subj_ctx_embds, std=0.02)
        self.subj_ctx_embds = nn.Parameter(subj_ctx_embds,requires_grad=True)  # to be optimized

        obj_ctx_embds = torch.empty(self.n_ctx, 768)
        nn.init.normal_(obj_ctx_embds, std=0.02)
        self.obj_ctx_embds = nn.Parameter(obj_ctx_embds,requires_grad=True)  # to be optimized

        if self.use_pos:
            self.relpos2embd = nn.Sequential(
                nn.Linear(12,256),
                nn.ReLU(),
                nn.Linear(256,512,bias=False)
            )
    
    def specify_clsids_range(self,split):
        if split == "base":
            pids_list = list(range(1,self.num_base+1))   # (1,2,...,92), len==92,   exclude __background__
        elif split == "novel":
            pids_list = list(range(self.num_base+1,self.num_base+self.num_novel+1))
            # (93,94,...,132), len == 40
        elif split == "all":
            pids_list = list(range(1,self.num_base+self.num_novel+1))    # len==132, i.e., 1 ~ 132
        else:
            assert False, "split must be base, novel, or all"
        
        return pids_list


    def forward(self,split):
        
        pids_list = self.specify_clsids_range(split.lower())

        n_cls = len(pids_list)
        prefix = self.prefix_embds[pids_list,:,:]
        suffix = self.suffix_embds[pids_list,:,:]
        token_mask = self.token_mask[pids_list,:]
        sub_ctx = self.subj_ctx_embds
        obj_ctx = self.obj_ctx_embds

        sub_ctx = sub_ctx.unsqueeze(0).expand(n_cls, -1, -1)
        obj_ctx = obj_ctx.unsqueeze(0).expand(n_cls, -1, -1)

        subj_token_embds = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)  # [CLS] token
                sub_ctx, # (n_cls, n_ctx, dim)  # context tokens
                suffix,  # (n_cls, *, dim)  # * refers to cls_name tokens, [SEP] token, and zero-pad tokens
            ],
            dim=1,
        )

        obj_token_embds = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                obj_ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

       
        return subj_token_embds,obj_token_embds,token_mask


class AlproPromptTrainer(nn.Module):

    def __init__(self,configs,is_train=True,train_on_gt_only=False):
        super().__init__()

        self.train_on_gt_only = train_on_gt_only
        self.num_base = configs["num_base"]
        self.num_novel = configs["num_novel"]
        self.temperature = configs["temperature"]  # alpro learned temperature
        n_context = configs["n_context_tokens"]
        pred_cls_split_info_path = configs["pred_cls_split_info_path"]

        alpro_model = setup_alpro_model()
        self.prompter = PromptLearner(n_context,alpro_model,pred_cls_split_info_path)
        self.text_encoder = AlproTextEncoder(alpro_model)


    def state_dict(self):

        return self.prompter.state_dict()
    
    def load_state_dict(self,state_dict):

        self.prompter.load_state_dict(state_dict)
    

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

        s_embds = F.normalize(s_embds,dim=-1)
        o_embds = F.normalize(o_embds,dim=-1)
        so_embds = torch.cat([s_embds,o_embds],dim=-1) / math.sqrt(2)  # (bsz,512)


        subj_token_embds,obj_token_embds,token_mask = self.prompter("base")    # (n_cls,max_L,768)
        subj_classifier_weights = self.text_encoder(subj_token_embds,token_mask)  # (n_cls,256)
        obj_classifier_weights = self.text_encoder(obj_token_embds,token_mask)
        
        classifier_weights = torch.cat([
            subj_classifier_weights,
            obj_classifier_weights
        ],dim=-1) / math.sqrt(2) # (n_cls, 512)

        relpos_embds = self.prompter.relpos2embd(relpos_feats)
        relpos_embds = F.normalize(relpos_embds,dim=-1)
        combined_embds = F.normalize(so_embds+relpos_embds,dim=-1)

        logits = torch.matmul(combined_embds,classifier_weights.t()) / self.temperature  # (bsz,n_cls)


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
            "total":loss.detach(),
        }
        return loss,loss_for_show

    
    def forward(self,batch_data,cls_split):
        if self.train_on_gt_only:
            return self.forward_on_gt_only(batch_data)
        
        print("warning !!!! using gt & det data is deprecated")
        (
            batch_det_feats,
            batch_traj_embds,
            batch_rel_pos_feat,
            batch_labels   
        )   = batch_data
        bsz = len(batch_traj_embds)

        batch_subj_embds = []
        batch_obj_embds = []
        for bid in range(bsz):
            traj_embds = batch_traj_embds[bid]
            n_det = traj_embds.shape[0]
            pair_ids = trajid2pairid(n_det).to(traj_embds.device)   # keep the same pair_id order as that in labels
            batch_subj_embds.append(traj_embds[pair_ids[:,0],:]) # (n_pair,2048)
            batch_obj_embds.append(traj_embds[pair_ids[:,1],:])
        
        batch_subj_embds = torch.cat(batch_subj_embds,dim=0)  # (N_pair,2048)
        batch_obj_embds = torch.cat(batch_obj_embds,dim=0)  # (N_pair,2048)
        batch_rel_pos_feat = torch.cat(batch_rel_pos_feat,dim=0)  # (N_pair,12)
        batch_labels = torch.cat(batch_labels,dim=0)  # shape == (N_pair,133)

        s_embds = F.normalize(batch_subj_embds,dim=-1)
        o_embds = F.normalize(batch_obj_embds,dim=-1)

        so_embds = torch.cat([s_embds,o_embds],dim=-1) / math.sqrt(2)  # (N_pair,512)

        
        subj_token_embds,obj_token_embds,token_mask,relpos_embds = self.prompter("base",batch_rel_pos_feat)    # (num_base,max_L,768)
        subj_classifier_weights = self.text_encoder(subj_token_embds,token_mask)  # (num_base,256)
        obj_classifier_weights = self.text_encoder(obj_token_embds,token_mask)    # has been normalized in text_encoder
        
        classifier_weights = torch.cat([
            subj_classifier_weights,
            obj_classifier_weights
        ],dim=-1) / math.sqrt(2)  # (num_base, 512)
        
        relpos_embds = F.normalize(relpos_embds,dim=-1)
        combined_embds = F.normalize(so_embds+relpos_embds,dim=-1)

        logits = torch.matmul(combined_embds,classifier_weights.t()) / self.temperature
        
        loss = self.loss_on_base(logits,batch_labels)
        
        return loss

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
            relpos_feat,
        )   = data


        n_det = traj_embds.shape[0]
        pair_ids = trajid2pairid(n_det).to(traj_embds.device)   # keep the same pair_id order as that in labels
        s_embds = traj_embds[pair_ids[:,0],:]  # (n_pair,256)
        o_embds = traj_embds[pair_ids[:,1],:]  # (n_pair,256)


        s_embds = F.normalize(s_embds,dim=-1)
        o_embds = F.normalize(o_embds,dim=-1)
    
        so_embds = torch.cat([s_embds,o_embds],dim=-1) / math.sqrt(2)  # (n_pair,512)
        relpos_embds = self.prompter.relpos2embd(relpos_feat)  # (n_pair,512)

        relpos_embds = F.normalize(relpos_embds,dim=-1)
        combined_embds = F.normalize(so_embds+relpos_embds,dim=-1)

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



##### conditional prompter (conditioned on subj& obj categories)


class PromptLearner_Conditional(nn.Module):

    def __init__(
        self,
        n_context,
        alpro_model,
        pred_cls_split_info_path,
        enti_text_emb_path = "/home/gkf/project/ALPRO/extract_features_output/vidvrd_ObjTextEmbeddings.npy",
        use_pos=True
    ):
        super().__init__()

        
        cls_split_info = load_json(pred_cls_split_info_path)
        self.num_base = sum([v=="base" for v in cls_split_info["cls2split"].values()])
        self.num_novel = sum([v=="novel" for v in cls_split_info["cls2split"].values()])
        
        tmp = np.load(enti_text_emb_path).astype('float32')    # __background__ is all 0's shape == (36,dim_emb)
        self.enti_txt_embds = nn.Parameter(torch.from_numpy(tmp),requires_grad=False)  # (36,dim_emb)


        cls2id_map = cls_split_info["cls2id"]
        cls_names = sorted(cls2id_map.items(),key= lambda x:x[1])
        cls_names = [x[0] for x in cls_names]  # including __background__
        cls_names = [name.replace("_", " ") for name in cls_names]
        name_lens = [len(name.split(" ")) for name in cls_names]
        self.n_cls = len(cls_names)
        self.n_ctx = n_context
        self.max_txt_len = GeneralAlproCfg.max_txt_len
        assert all([len_ + self.n_ctx <= self.max_txt_len for len_ in name_lens])

        place_holder_strs = " ".join(["X"] * self.n_ctx)
        token_strs = [place_holder_strs + " " + name for name in cls_names] # including __background__

        tokenizer = BertTokenizerFast.from_pretrained("/home/gkf/project/ALPRO/ext/bert-base-uncased")
        batch_enc = tokenizer.batch_encode_plus(
            token_strs,  # bsz= n_cls
            max_length= self.max_txt_len,  # default: 40
            padding='max_length',
            return_tensors="pt",
            truncation=True
        )
        token_ids = batch_enc.input_ids  # (n_cls, max_L)  # max_L == 40
        token_mask = batch_enc.attention_mask  # (n_cls, max_L)
        # [CLS] t1 t2 ... tn [SEP] 0, 0, 0 ..., 0 (zero-padding to max_L == 40) refer to `tools/token_ids.png`

        with torch.no_grad():
            token_embds =  alpro_model.text_encoder.bert.embeddings.word_embeddings(token_ids)  # (n_cls,40,768), n_cls  including __background__
        
        prefix_embds =  token_embds[:, :1, :]  # (n_cls,1,768), [CLS] token embedding
        suffix_embds =  token_embds[:, 1 + self.n_ctx :, :]  # (n_cls, 40-1-n_ctx ,768) embedding of cls_name tokens, [SEP] token, and zero-pad tokens

        self.register_buffer("prefix_embds", prefix_embds)
        self.register_buffer("suffix_embds", suffix_embds)
        self.register_buffer("token_mask",token_mask)


        meta_ctx_embds = torch.empty(self.n_ctx, 768)
        nn.init.normal_(meta_ctx_embds, std=0.02)
        self.meta_ctx_embds = nn.Parameter(meta_ctx_embds,requires_grad=True)  # to be optimized

        subj_ctx_embds = torch.empty(self.n_ctx, 768)
        nn.init.normal_(subj_ctx_embds, std=0.02)
        self.subj_ctx_embds = nn.Parameter(subj_ctx_embds,requires_grad=True)  # to be optimized

        obj_ctx_embds = torch.empty(self.n_ctx, 768)
        nn.init.normal_(obj_ctx_embds, std=0.02)
        self.obj_ctx_embds = nn.Parameter(obj_ctx_embds,requires_grad=True)  # to be optimized

        self.meta_cls2embd = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,768*2,bias=False)
        )

        if use_pos:
            self.relpos2embd = nn.Sequential(
                nn.Linear(12,256),
                nn.ReLU(),
                nn.Linear(256,512,bias=False)
            )
    
    def specify_clsids_range(self,split):
        if split == "base":
            pids_list = list(range(1,self.num_base+1))   # (1,2,...,92), len==92,   exclude __background__
        elif split == "novel":
            pids_list = list(range(self.num_base+1,self.num_base+self.num_novel+1))
            # (93,94,...,132), len == 40
        elif split == "all":
            pids_list = list(range(1,self.num_base+self.num_novel+1))    # len==132, i.e., 1 ~ 132
        else:
            assert False, "split must be base, novel, or all"
        
        return pids_list


    def forward(self,split,so_cls_ids):
        n_pair = so_cls_ids.shape[0]
        s_embd = self.enti_txt_embds[so_cls_ids[:,0],:] # (n_pair,256)
        o_embd = self.enti_txt_embds[so_cls_ids[:,1],:] # (n_pair,256)
        so_embds = torch.cat([s_embd,o_embd],dim=-1)  # (n_pair,512)
        so_embds = self.meta_cls2embd(so_embds)  # (n_pair,768*2)
        
        s_q,o_q = so_embds[:,:768],so_embds[:,768:]  # (n_pair,768)
        k = v = self.meta_ctx_embds  # (n_ctx,768)
        s_ctx = F.softmax(s_q @ k.t()/math.sqrt(768),dim=-1) @ v  # (n_pair,768)
        o_ctx = F.softmax(o_q @ k.t()/math.sqrt(768),dim=-1) @ v  # (n_pair,768)

        s_ctx = self.subj_ctx_embds[None,:,:] + s_ctx[:,None,:] # (n_pair,n_ctx,768)
        o_ctx = self.obj_ctx_embds[None,:,:] + o_ctx[:,None,:]  # (n_pair,n_ctx,768)

        
        pids_list = self.specify_clsids_range(split.lower())

        n_cls = len(pids_list)
        prefix = self.prefix_embds[pids_list,:,:][None,:,:,:].expand(n_pair,-1,-1,-1) # (n_pair,n_cls,1,dim)
        suffix = self.suffix_embds[pids_list,:,:][None,:,:,:].expand(n_pair,-1,-1,-1) # (n_pair,n_cls,*,dim), * = max_L - 1 - n_ctx
        token_mask = self.token_mask[pids_list,:]  # (n_cls,max_L)
        max_L = token_mask.shape[-1]

        s_ctx = s_ctx[:,None,:,:].expand(-1,n_cls, -1, -1)  # (n_pair,n_cls,n_ctx,768)
        o_ctx = o_ctx[:,None,:,:].expand(-1,n_cls, -1, -1)

        subj_token_embds = torch.cat(
            [
                prefix,  # (n_pair, n_cls, 1, dim)  # [CLS] token
                s_ctx,   # (n_pair, n_cls, n_ctx, dim)  # context tokens
                suffix,  # (n_pair, n_cls, *, dim)  # * refers to cls_name tokens, [SEP] token, and zero-pad tokens
            ],
            dim=2,
        )  # (n_pair,n_cls,max_L,768)

        obj_token_embds = torch.cat(
            [
                prefix,  #
                o_ctx, #
                suffix,  # 
            ],
            dim=2,
        )

        subj_token_embds = subj_token_embds.reshape(n_pair*n_cls,max_L,768)
        obj_token_embds = obj_token_embds.reshape(n_pair*n_cls,max_L,768)
        token_mask = token_mask.repeat(n_pair,1)  # (n_pair*n_cls,max_L)  
        # refer to `test_API/test_repeat.py` for repeat vs. repeat_interleave, 
        # here using repeat is correct and  repeat_interleave is wrong
        # because in broadcast shape (n_pair,n_cls,max_L,768), n_pair is the first dim

 
        return subj_token_embds,obj_token_embds,token_mask


class PromptLearner_Conditional_v2(nn.Module):
    '''
    conditioned on subj/obj categories separately (decoupled)
    '''

    def __init__(
        self,
        n_context,
        alpro_model,
        pred_cls_split_info_path,
        enti_text_emb_path = "/home/gkf/project/ALPRO/extract_features_output/vidvrd_ObjTextEmbeddings.npy",
        use_pos=True
    ):
        super().__init__()

        
        cls_split_info = load_json(pred_cls_split_info_path)
        self.num_base = sum([v=="base" for v in cls_split_info["cls2split"].values()])
        self.num_novel = sum([v=="novel" for v in cls_split_info["cls2split"].values()])
        
        tmp = np.load(enti_text_emb_path).astype('float32')    # __background__ is all 0's shape == (36,dim_emb)
        self.enti_txt_embds = nn.Parameter(torch.from_numpy(tmp),requires_grad=False)  # (36,dim_emb)


        cls2id_map = cls_split_info["cls2id"]
        cls_names = sorted(cls2id_map.items(),key= lambda x:x[1])
        cls_names = [x[0] for x in cls_names]  # including __background__
        cls_names = [name.replace("_", " ") for name in cls_names]
        name_lens = [len(name.split(" ")) for name in cls_names]
        self.n_cls = len(cls_names)
        self.n_ctx = n_context
        self.max_txt_len = GeneralAlproCfg.max_txt_len
        assert all([len_ + self.n_ctx <= self.max_txt_len for len_ in name_lens])

        place_holder_strs = " ".join(["X"] * self.n_ctx)
        token_strs = [place_holder_strs + " " + name for name in cls_names] # including __background__

        tokenizer = BertTokenizerFast.from_pretrained("/home/gkf/project/ALPRO/ext/bert-base-uncased")
        batch_enc = tokenizer.batch_encode_plus(
            token_strs,  # bsz= n_cls
            max_length= self.max_txt_len,  # default: 40
            padding='max_length',
            return_tensors="pt",
            truncation=True
        )
        token_ids = batch_enc.input_ids  # (n_cls, max_L)  # max_L == 40
        token_mask = batch_enc.attention_mask  # (n_cls, max_L)
        # [CLS] t1 t2 ... tn [SEP] 0, 0, 0 ..., 0 (zero-padding to max_L == 40) refer to `tools/token_ids.png`

        with torch.no_grad():
            token_embds =  alpro_model.text_encoder.bert.embeddings.word_embeddings(token_ids)  # (n_cls,40,768), n_cls  including __background__
        
        prefix_embds =  token_embds[:, :1, :]  # (n_cls,1,768), [CLS] token embedding
        suffix_embds =  token_embds[:, 1 + self.n_ctx :, :]  # (n_cls, 40-1-n_ctx ,768) embedding of cls_name tokens, [SEP] token, and zero-pad tokens

        self.register_buffer("prefix_embds", prefix_embds)
        self.register_buffer("suffix_embds", suffix_embds)
        self.register_buffer("token_mask",token_mask)

        subj_ctx_embds = torch.empty(self.n_ctx, 768)
        nn.init.normal_(subj_ctx_embds, std=0.02)
        self.subj_ctx_embds = nn.Parameter(subj_ctx_embds,requires_grad=True)  # to be optimized

        obj_ctx_embds = torch.empty(self.n_ctx, 768)
        nn.init.normal_(obj_ctx_embds, std=0.02)
        self.obj_ctx_embds = nn.Parameter(obj_ctx_embds,requires_grad=True)  # to be optimized

        self.meta_subjcls2embd = nn.Sequential(
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,768,bias=False)
        )
        self.meta_objcls2embd = nn.Sequential(
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,768,bias=False)
        )

        if use_pos:
            self.relpos2embd = nn.Sequential(
                nn.Linear(12,256),
                nn.ReLU(),
                nn.Linear(256,512,bias=False)
            )
    
    def specify_clsids_range(self,split):
        if split == "base":
            pids_list = list(range(1,self.num_base+1))   # (1,2,...,92), len==92,   exclude __background__
        elif split == "novel":
            pids_list = list(range(self.num_base+1,self.num_base+self.num_novel+1))
            # (93,94,...,132), len == 40
        elif split == "all":
            pids_list = list(range(1,self.num_base+self.num_novel+1))    # len==132, i.e., 1 ~ 132
        else:
            assert False, "split must be base, novel, or all"
        
        return pids_list


    def forward(self,split,so_cls_ids):
        n_pair = so_cls_ids.shape[0]
        s_embd = self.enti_txt_embds[so_cls_ids[:,0],:] # (n_pair,256)
        o_embd = self.enti_txt_embds[so_cls_ids[:,1],:] # (n_pair,256)
        
        s_embd = self.meta_subjcls2embd(s_embd)  # (n_pair,768)
        o_embd = self.meta_objcls2embd(o_embd)

        s_ctx = self.subj_ctx_embds[None,:,:] + s_embd[:,None,:] # (n_pair,n_ctx,768)
        o_ctx = self.obj_ctx_embds[None,:,:] + o_embd[:,None,:]  # (n_pair,n_ctx,768)

        
        pids_list = self.specify_clsids_range(split.lower())

        n_cls = len(pids_list)
        prefix = self.prefix_embds[pids_list,:,:][None,:,:,:].expand(n_pair,-1,-1,-1) # (n_pair,n_cls,1,dim)
        suffix = self.suffix_embds[pids_list,:,:][None,:,:,:].expand(n_pair,-1,-1,-1) # (n_pair,n_cls,*,dim), * = max_L - 1 - n_ctx
        token_mask = self.token_mask[pids_list,:]  # (n_cls,max_L)
        max_L = token_mask.shape[-1]

        s_ctx = s_ctx[:,None,:,:].expand(-1,n_cls, -1, -1)  # (n_pair,n_cls,n_ctx,768)
        o_ctx = o_ctx[:,None,:,:].expand(-1,n_cls, -1, -1)

        subj_token_embds = torch.cat(
            [
                prefix,  # (n_pair, n_cls, 1, dim)  # [CLS] token
                s_ctx,   # (n_pair, n_cls, n_ctx, dim)  # context tokens
                suffix,  # (n_pair, n_cls, *, dim)  # * refers to cls_name tokens, [SEP] token, and zero-pad tokens
            ],
            dim=2,
        )  # (n_pair,n_cls,max_L,768)

        obj_token_embds = torch.cat(
            [
                prefix,  #
                o_ctx, #
                suffix,  # 
            ],
            dim=2,
        )

        subj_token_embds = subj_token_embds.reshape(n_pair*n_cls,max_L,768)
        obj_token_embds = obj_token_embds.reshape(n_pair*n_cls,max_L,768)
        token_mask = token_mask.repeat(n_pair,1)  # (n_pair*n_cls,max_L)  
        # refer to `test_API/test_repeat.py` for repeat vs. repeat_interleave, 
        # here using repeat is correct and  repeat_interleave is wrong
        # because in broadcast shape (n_pair,n_cls,max_L,768), n_pair is the first dim

        return subj_token_embds,obj_token_embds,token_mask

class AlproPromptTrainer_Conditional(nn.Module):
    # refer to `test_API/test_metaNet.py`
    def __init__(self,configs,is_train=True,train_on_gt_only=False):
        super().__init__()

        self.train_on_gt_only = train_on_gt_only
        self.num_base = configs["num_base"]
        self.num_novel = configs["num_novel"]
        self.temperature = configs["temperature"]  # alpro learned temperature
        n_context = configs["n_context_tokens"]
        pred_cls_split_info_path = configs["pred_cls_split_info_path"]
        self.conditioned_on_enti_cls = True

        alpro_model = setup_alpro_model()
        self.prompter = PromptLearner_Conditional(n_context,alpro_model,pred_cls_split_info_path)
        self.text_encoder = AlproTextEncoder(alpro_model)


    def state_dict(self):

        return self.prompter.state_dict()
    
    def load_state_dict(self,state_dict):

        self.prompter.load_state_dict(state_dict)
    

    def forward(self,batch_data,cls_split):
        assert self.train_on_gt_only
        (
            s_roi_feats,      # (bsz,2048)
            o_roi_feats,
            s_embds,          # (bsz,256)
            o_embds,
            relpos_feats,     # (bsz,12)
            triplet_cls_ids   # list[tensor] each shape == (n_preds,3)
        ) = batch_data
        

        bsz = len(triplet_cls_ids)  # bsz == n_pair
        multihot = torch.zeros(size=(bsz,self.num_base),device=s_embds.device)
        so_cls_ids = []
        for i in range(bsz):
            spo_cls_ids = triplet_cls_ids[i]   # (n_pred,3)
            p_clsids = spo_cls_ids[:,1] - 1  # (n_pred,)  
            # range: 1 ~ num_base  --> 0 ~ num_base -1
            # NOTE this has filtered in dataloader
            multihot[i,p_clsids] = 1

            so_cls_ids.append(spo_cls_ids[0,[0,2]])  # (2,)
        so_cls_ids = torch.stack(so_cls_ids,dim=0)  # (n_pair,2)

        s_embds = F.normalize(s_embds,dim=-1)
        o_embds = F.normalize(o_embds,dim=-1)
        so_embds = torch.cat([s_embds,o_embds],dim=-1) / math.sqrt(2)  # (bsz,512)

        uniq_so_cls_ids,index_map = unique_with_idx_nd(so_cls_ids) # (n_uniq,2)
        n_uniq = uniq_so_cls_ids.shape[0]

        subj_token_embds,obj_token_embds,token_mask = self.prompter("base",uniq_so_cls_ids)    # (n_uniq*n_cls,max_L,768), mask.shape == (n_uniq*n_cls,max_L)
        subj_classifier_weights = self.text_encoder(subj_token_embds,token_mask)  # (n_uniq*n_cls,256)
        obj_classifier_weights = self.text_encoder(obj_token_embds,token_mask)     
        
        classifier_weights = torch.cat([
            subj_classifier_weights,
            obj_classifier_weights
        ],dim=-1) / math.sqrt(2) # (n_uniq*n_cls, 512)
        classifier_weights = classifier_weights.reshape(n_uniq,-1,512)  # (n_uniq,n_cls,512)

        relpos_embds = self.prompter.relpos2embd(relpos_feats)  # (n_pair,512)
        relpos_embds = F.normalize(relpos_embds,dim=-1)
        combined_embds = F.normalize(so_embds+relpos_embds,dim=-1) # (n_pair,512)
        combined_embds = combined_embds[:,:,None]  # (n_pair,512,1)

        ### use index_map to re-construct n_pair so_cls_ids
        repeat_lens = torch.as_tensor([len(im) for im in index_map],device=combined_embds.device)
        index_all = torch.cat(index_map,dim=0)
        classifier_weights = classifier_weights.repeat_interleave(repeat_lens,dim=0)  # (n_pair,n_cls,512)
        combined_embds = combined_embds[index_all,:,:]


        logits = torch.bmm(classifier_weights,combined_embds) / self.temperature 
        # (n_pair,n_cls,512) x (n_pair,512,1) --> (n_pair,n_cls,1)
        logits = logits.squeeze(2)  # (n_pair,n_cls)


        loss = sigmoid_focal_loss(logits,multihot,reduction='mean')

        loss_for_show = {
            "total":loss.detach(),
        }
        return loss,loss_for_show


    def reset_classifier_weights(self,cls_split):
        if self.training:
            return
        # this func is used in test , reset for each epoch
        # for each epoch， reset once and save the classifier_weights as buffer,
        # reset at each iteration is not necessary and is too time consuming
        # and we must re-reset for each epoch
        
        num_enti_cls = self.prompter.enti_txt_embds.shape[0] # 36, include __background__
        enti_cls_ids = torch.as_tensor(range(1,num_enti_cls))  # 1~35
        total_so_clsids = torch.cartesian_prod(enti_cls_ids,enti_cls_ids)  # (35*35, 2)

        totoal_pair = total_so_clsids.shape[0]  # 1225
        bsz = 16 
        total_classifier_weights = []
        for i in range(0,totoal_pair,bsz):  # use for-loop to avoid CUDA out of memory
            so_clsids = total_so_clsids[i:i+bsz,...]
            bsz_ = so_clsids.shape[0]  # this can be < bsz
            
            subj_token_embds,obj_token_embds,token_mask = self.prompter(cls_split,so_clsids)    # (bsz*n_cls,max_L,768)
            subj_classifier_weights = self.text_encoder(subj_token_embds,token_mask)  # (bsz*n_cls,256)
            obj_classifier_weights = self.text_encoder(obj_token_embds,token_mask)
            
            classifier_weights = torch.cat([
                subj_classifier_weights,
                obj_classifier_weights
            ],dim=-1)  / math.sqrt(2)   # (bsz*n_cls, 512)

            classifier_weights = classifier_weights.reshape(bsz_,-1,512)  # (bsz,n_cls,512)
            total_classifier_weights.append(classifier_weights)
        total_classifier_weights = torch.cat(total_classifier_weights,dim=0) # (totoal_pair,n_cls,512), totoal_pair = 1225=35*35


        self.register_buffer("total_classifier_weights",total_classifier_weights,persistent=False)
        total_classifier_weights.tolist()
    
    def so_to_relative_ids_map(self,so_clsids):
        '''
        map each pair subj/obj cls_id to a unique id
        total range: 0~1224, 35*35=1225
        e.g., (1,1)-->0; (1,2)-->1; ... (35,34)-->1223; (35,35)-->1225
        refer to `test_API/so_to_relative_ids_map.py`
        '''
        # so_clsids.shape == (n_pair,2)
        num_enti_cls_wo_bg = self.prompter.enti_txt_embds.shape[0] -1 # 35
        relative_ids = (so_clsids[:,0]-1)*num_enti_cls_wo_bg + so_clsids[:,1]-1

        return relative_ids

    def forward_inference_bsz1_backup(self,data,cls_split,pred_topk=10):
        # bsz1 means 1 segment
        # this func has been deprecated, it has to calculate classifier_weight at each iteration for each unique pair
        (
            det_feats,
            traj_embds,
            relpos_feat,
            traj_cls_ids  # (n_det,)
        )   = data


        n_det = traj_embds.shape[0]
        pair_ids = trajid2pairid(n_det).to(traj_embds.device)   # keep the same pair_id order as that in labels
        sids = pair_ids[:,0]
        oids = pair_ids[:,1]
        s_embds = traj_embds[sids,:]  # (n_pair,256)
        o_embds = traj_embds[oids,:]  # (n_pair,256)
        s_embds = F.normalize(s_embds,dim=-1)
        o_embds = F.normalize(o_embds,dim=-1)
    
        so_embds = torch.cat([s_embds,o_embds],dim=-1) / math.sqrt(2)  # (n_pair,512)
        relpos_embds = self.prompter.relpos2embd(relpos_feat)  # (n_pair,512)

        relpos_embds = F.normalize(relpos_embds,dim=-1)
        combined_embds = F.normalize(so_embds+relpos_embds,dim=-1)  # (n_pair,512)

        so_cls_ids = torch.stack([traj_cls_ids[sids],traj_cls_ids[oids]],dim=-1)  # (n_pair,2)
        uniq_so_cls_ids,index_map = unique_with_idx_nd(so_cls_ids) # (n_uniq,2)
        n_uniq = uniq_so_cls_ids.shape[0]

        subj_token_embds,obj_token_embds,token_mask = self.prompter(cls_split,uniq_so_cls_ids)    # (n_uniq*n_cls,max_L,768)
        subj_classifier_weights = self.text_encoder(subj_token_embds,token_mask)  # (n_uniq*n_cls,256)
        obj_classifier_weights = self.text_encoder(obj_token_embds,token_mask)    # has been normalized in text_encoder
        
        classifier_weights = torch.cat([
            subj_classifier_weights,
            obj_classifier_weights
        ],dim=-1) / math.sqrt(2)  # (n_uniq*n_cls, 512)
        classifier_weights = classifier_weights.reshape(n_uniq,-1,512)  # (n_uniq,n_cls,512)

        repeat_lens = torch.as_tensor([len(im) for im in index_map],device=classifier_weights.device)
        index_all = torch.cat(index_map,dim=0)
        classifier_weights = classifier_weights.repeat_interleave(repeat_lens,dim=0)  # (n_pair,n_cls,512)
        combined_embds = combined_embds[index_all,:,None]  # (n_pair,512,1)
        pair_ids = pair_ids[index_all,:]


        logits = torch.bmm(classifier_weights,combined_embds) / self.temperature 
        # (n_pair,n_cls,512) x (n_pair,512,1) --> (n_pair,n_cls,1)
        logits = logits.squeeze(2)  # (n_pair,n_cls)

     
        pred_probs = torch.sigmoid(logits)  # (n_pair, num_cls) , exclude __background__
        scores,cls_ids = _to_predict_cls_ids(  # (n_pair,k)
            cls_split,
            self.num_base,
            self.num_novel,
            pred_probs,
            pred_topk,
        )
        

        return scores,cls_ids,pair_ids


    def forward_inference_bsz1(self,data,cls_split,pred_topk=10):
        # bsz1 means 1 segment
        # this func has been deprecated, it has to calculate classifier_weight at each iteration for each unique pair
        (
            det_feats,
            traj_embds,
            relpos_feat,
            traj_cls_ids  # (n_det,)
        )   = data


        n_det = traj_embds.shape[0]
        pair_ids = trajid2pairid(n_det).to(traj_embds.device)   # keep the same pair_id order as that in labels
        sids = pair_ids[:,0]
        oids = pair_ids[:,1]
        s_embds = traj_embds[sids,:]  # (n_pair,256)
        o_embds = traj_embds[oids,:]  # (n_pair,256)
        s_embds = F.normalize(s_embds,dim=-1)
        o_embds = F.normalize(o_embds,dim=-1)
    
        so_embds = torch.cat([s_embds,o_embds],dim=-1) / math.sqrt(2)  # (n_pair,512)
        relpos_embds = self.prompter.relpos2embd(relpos_feat)  # (n_pair,512)

        relpos_embds = F.normalize(relpos_embds,dim=-1)
        combined_embds = F.normalize(so_embds+relpos_embds,dim=-1)  # (n_pair,512)
        combined_embds = combined_embds[:,:,None] # (n_pair,512,1)

        so_clsids = torch.stack([traj_cls_ids[sids],traj_cls_ids[oids]],dim=-1)  # (n_pair,2)
        so_relids = self.so_to_relative_ids_map(so_clsids)  # (n_pair,)
        classifier_weights = self.total_classifier_weights[so_relids,:,:]  # (n_pair,n_cls,512)


        logits = torch.bmm(classifier_weights,combined_embds) / self.temperature 
        # (n_pair,n_cls,512) x (n_pair,512,1) --> (n_pair,n_cls,1)
        logits = logits.squeeze(2)  # (n_pair,n_cls)

     
        pred_probs = torch.sigmoid(logits)  # (n_pair, num_cls) , exclude __background__
        scores,cls_ids = _to_predict_cls_ids(  # (n_pair,k)
            cls_split,
            self.num_base,
            self.num_novel,
            pred_probs,
            pred_topk,
        )
        

        return scores,cls_ids,pair_ids


class AlproPromptTrainer_Conditional_v2(AlproPromptTrainer_Conditional):
    def __init__(self,configs,is_train=True,train_on_gt_only=False):
        nn.Module.__init__(self)

        self.train_on_gt_only = train_on_gt_only
        self.num_base = configs["num_base"]
        self.num_novel = configs["num_novel"]
        self.temperature = configs["temperature"]  # alpro learned temperature
        n_context = configs["n_context_tokens"]
        pred_cls_split_info_path = configs["pred_cls_split_info_path"]
        self.conditioned_on_enti_cls = True

        alpro_model = setup_alpro_model()
        self.prompter = PromptLearner_Conditional_v2(n_context,alpro_model,pred_cls_split_info_path)
        self.text_encoder = AlproTextEncoder(alpro_model)


####### grouped prompt (according to relative GIoU of subj/obj)

def get_giou_tags_v0(rel_gious,giou_th):
    '''
    FIXME BUG 这个函数只有当 rel_gious 包含所有六种情况时，返回的prompt_id才是对的
    '''
    s_tags = rel_gious[:,0] >= giou_th  # (n_pair,)
    e_tags = rel_gious[:,1] >= giou_th
    diff_tags = (rel_gious[:,1] - rel_gious[:,0]) >= 0  # (n_pair,)

    giou_tags = torch.stack([s_tags,e_tags,diff_tags],dim=-1)  # (n_pair,3)

    # uniq_tags,count = torch.unique(giou_tags,return_counts=True,dim=0)
    uniq_tags,inverse_ids,count = torch.unique(giou_tags,return_counts=True,return_inverse=True,dim=0,sorted=True)
    assert len(count) == 6
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

def get_giou_tags(rel_gious,giou_th):

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

    uniq_tags,inverse_ids,count = torch.unique(giou_tags_,return_counts=True,return_inverse=True,dim=0,sorted=True)
    assert len(count) == 6
    inverse_ids = inverse_ids[6:]
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



    uniq_tags,inverse_ids,count = torch.unique(giou_tags_,return_counts=True,return_inverse=True,dim=0,sorted=True)
    print(uniq_tags,count)
    inverse_ids = inverse_ids[6:]

class PromptLearner_Grouped(PromptLearner):
    def __init__(self, n_groups,n_context, alpro_model, cls_split_info_path, use_pos=True):
        self.n_groups = n_groups
        super().__init__(n_context, alpro_model, cls_split_info_path, use_pos)
        # the `__init__` of Father will call the `setup_learnable_parameters` of Child, refer to `test_API/test_Class.py`


    def setup_learnable_parameters(self):
        
        # self.subj_ctx_embds = nn.ParameterList()
        # self.obj_ctx_embds = nn.ParameterList()

        # for i in range(self.n_groups):
        #     subj_ctx_embds = torch.empty(self.n_ctx, 768)
        #     nn.init.normal_(subj_ctx_embds, std=0.02)
        #     self.subj_ctx_embds.append(
        #         nn.Parameter(subj_ctx_embds,requires_grad=True)
        #     )
        
        #     obj_ctx_embds = torch.empty(self.n_ctx, 768)
        #     nn.init.normal_(obj_ctx_embds, std=0.02)
        #     self.obj_ctx_embds.append(
        #         nn.Parameter(obj_ctx_embds,requires_grad=True)  # to be optimized
        #     )

        subj_ctx_embds = torch.empty(self.n_groups,self.n_ctx, 768)
        nn.init.normal_(subj_ctx_embds, std=0.02)
        self.subj_ctx_embds = nn.Parameter(subj_ctx_embds,requires_grad=True)

        obj_ctx_embds = torch.empty(self.n_groups,self.n_ctx, 768)
        nn.init.normal_(obj_ctx_embds, std=0.02)
        self.obj_ctx_embds = nn.Parameter(obj_ctx_embds,requires_grad=True)


        if self.use_pos:
            self.relpos2embd = nn.Sequential(
                nn.Linear(12,256),
                nn.ReLU(),
                nn.Linear(256,512,bias=False)
            )


    def forward(self,split):
        n_grp = self.n_groups
        s_ctx = self.subj_ctx_embds  # (n_grp,n_ctx,768)
        o_ctx = self.obj_ctx_embds   # (n_grp,n_ctx,768)
        
        pids_list = self.specify_clsids_range(split.lower())

        n_cls = len(pids_list)
        prefix = self.prefix_embds[pids_list,:,:][None,:,:,:].expand(n_grp,-1,-1,-1) # (n_grp,n_cls,1,dim)
        suffix = self.suffix_embds[pids_list,:,:][None,:,:,:].expand(n_grp,-1,-1,-1) # (n_grp,n_cls,*,dim), * = max_L - 1 - n_ctx
        token_mask = self.token_mask[pids_list,:]  # (n_cls,max_L)
        max_L = token_mask.shape[-1]

        s_ctx = s_ctx[:,None,:,:].expand(-1,n_cls, -1, -1)  # (n_grp,n_cls,n_ctx,768)
        o_ctx = o_ctx[:,None,:,:].expand(-1,n_cls, -1, -1)

        subj_token_embds = torch.cat(
            [
                prefix,  # (n_grp, n_cls, 1, dim)  # [CLS] token
                s_ctx,   # (n_grp, n_cls, n_ctx, dim)  # context tokens
                suffix,  # (n_grp, n_cls, *, dim)  # * refers to cls_name tokens, [SEP] token, and zero-pad tokens
            ],
            dim=2,
        )  # (n_group,n_cls,max_L,768)

        obj_token_embds = torch.cat(
            [
                prefix,  #
                o_ctx, #
                suffix,  # 
            ],
            dim=2,
        )

        subj_token_embds = subj_token_embds.reshape(n_grp*n_cls,max_L,768)
        obj_token_embds = obj_token_embds.reshape(n_grp*n_cls,max_L,768)
        token_mask = token_mask.repeat(n_grp,1)  # (n_grp*n_cls,max_L)  
        # refer to `test_API/test_repeat.py` for repeat vs. repeat_interleave, 
        # here using repeat is correct and  repeat_interleave is wrong
        # because in broadcast shape (n_grp,n_cls,max_L,768), n_pair is the first dim

 
        return subj_token_embds,obj_token_embds,token_mask


class AlproPromptTrainer_Grouped(nn.Module):

    def __init__(self,configs,is_train=True,train_on_gt_only=False):
        super().__init__()

        self.train_on_gt_only = train_on_gt_only
        self.fullysupervise = configs.get("fullysupervise",False)
        self.num_base = configs["num_base"]
        self.num_novel = configs["num_novel"]
        self.temperature = configs["temperature"]  # alpro learned temperature
        self.n_context = configs["n_context_tokens"]
        self.n_groups  =configs["n_prompt_groups"]
        self.giou_th = configs["giou_th"]
        pred_cls_split_info_path = configs["pred_cls_split_info_path"]

        alpro_model = setup_alpro_model()
        self.prompter = PromptLearner_Grouped(self.n_groups,self.n_context,alpro_model,pred_cls_split_info_path)
        self.text_encoder = AlproTextEncoder(alpro_model)


    def state_dict(self):

        return self.prompter.state_dict()
    
    def load_state_dict(self,state_dict):

        self.prompter.load_state_dict(state_dict)


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

        uniq_tags,inverse_ids,count = torch.unique(giou_tags_,return_counts=True,return_inverse=True,dim=0,sorted=True)
        assert len(count) == 6
        inverse_ids = inverse_ids[6:]
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

        uniq_tags,inverse_ids,count = torch.unique(giou_tags_,return_counts=True,return_inverse=True,dim=0,sorted=True)
        print(uniq_tags,count)
        inverse_ids = inverse_ids[6:]


    def forward_on_gt_only(self,batch_data):
        (
            s_roi_feats,      # (bsz,2048)
            o_roi_feats,
            s_embds,          # (bsz,256)
            o_embds,
            relpos_feats,     # (bsz,12)
            rel_gious,         # (bsz,2)
            triplet_cls_ids   # list[tensor] each shape == (n_preds,3)
        ) = batch_data
        

        bsz = len(triplet_cls_ids)  # bsz == n_pair
        n_grp = self.n_groups

        s_embds = F.normalize(s_embds,dim=-1)
        o_embds = F.normalize(o_embds,dim=-1)
        so_embds = torch.cat([s_embds,o_embds],dim=-1) / math.sqrt(2)  # (bsz,512)

        _cls_split = "all" if self.fullysupervise else "base"
            
        subj_token_embds,obj_token_embds,token_mask = self.prompter(_cls_split)    # (n_grp*n_cls,max_L,768), mask.shape == (n_grp*n_cls,max_L)
        subj_classifier_weights = self.text_encoder(subj_token_embds,token_mask)  # (n_grp*n_cls,256)
        obj_classifier_weights = self.text_encoder(obj_token_embds,token_mask)
        
        classifier_weights = torch.cat([
            subj_classifier_weights,
            obj_classifier_weights
        ],dim=-1) / math.sqrt(2) # (n_grp*n_cls, 512)
        classifier_weights = classifier_weights.reshape(n_grp,-1,512)  # (n_grp,n_cls,512)

        giou_tags,prompt_ids,_ = self.get_giou_tags(rel_gious,self.giou_th)  
        # giou_tags.shape == (bsz,3) each tag is a 3-dim binary vector
        # prompt_ids.shape == (bsz,)
        classifier_weights = classifier_weights[prompt_ids,:,:]  # (bsz,n_cls,512)

        relpos_embds = self.prompter.relpos2embd(relpos_feats)  # (bsz,512)
        relpos_embds = F.normalize(relpos_embds,dim=-1)
        combined_embds = F.normalize(so_embds+relpos_embds,dim=-1)  # (bsz,512)
        combined_embds = combined_embds[:,:,None]  # (bsz,512,1)

        logits = torch.bmm(classifier_weights,combined_embds) / self.temperature 
        # (bsz,n_cls,512) x (bsz,512,1) --> (bsz,n_cls,1)
        logits = logits.squeeze(2)  # (bsz,n_cls)

        _ncls = self.num_base + self.num_novel if self.fullysupervise else self.num_base
        multihot = torch.zeros(size=(bsz,_ncls),device=logits.device)
        for i in range(bsz):
            spo_cls_ids = triplet_cls_ids[i]   # (n_pred,3)
            p_clsids = spo_cls_ids[:,1] - 1  # (n_pred,)  
            # range: 1 ~ num_base  --> 0 ~ num_base -1
            # NOTE this has filtered in dataloader
            multihot[i,p_clsids] = 1

        loss = sigmoid_focal_loss(logits,multihot,reduction='mean')

        loss_for_show = {
            "total":loss.detach(),
        }
        return loss,loss_for_show

    
    def forward(self,batch_data,cls_split):
        if self.train_on_gt_only:
            return self.forward_on_gt_only(batch_data)
        
        # TODO
        return loss


    def reset_classifier_weights(self,cls_split):

        if self.training:
            return
        # this func is used in test , reset for each epoch
        # for each epoch， reset once and save the classifier_weights as buffer,
        # reset at each iteration is not necessary and is too time consuming
        # and we must re-reset for each epoch
        

        subj_token_embds,obj_token_embds,token_mask = self.prompter(cls_split)    # (n_grp*n_cls,max_L,768)
        subj_classifier_weights = self.text_encoder(subj_token_embds,token_mask)  # (n_grp*n_cls,256)
        obj_classifier_weights = self.text_encoder(obj_token_embds,token_mask)
        
        classifier_weights = torch.cat([
            subj_classifier_weights,
            obj_classifier_weights
        ],dim=-1)  / math.sqrt(2)   # (n_grp*n_cls, 512)

        classifier_weights = classifier_weights.reshape(self.n_groups,-1,512)  # (bsz,n_cls,512)
        
        self.register_buffer("group_classifier_weights",classifier_weights,persistent=False)


    def forward_inference_bsz1(self,data,cls_split,pred_topk=10):
        # bsz1 means 1 segment
        (
            det_feats,
            traj_embds,
            rel_pos_and_giou,
        )   = data

        relpos_feat,rel_gious = rel_pos_and_giou

        n_det = traj_embds.shape[0]
        pair_ids = trajid2pairid(n_det).to(traj_embds.device)   # keep the same pair_id order as that in labels
        s_embds = traj_embds[pair_ids[:,0],:]  # (n_pair,256)
        o_embds = traj_embds[pair_ids[:,1],:]  # (n_pair,256)
        s_embds = F.normalize(s_embds,dim=-1)
        o_embds = F.normalize(o_embds,dim=-1)
    
        so_embds = torch.cat([s_embds,o_embds],dim=-1) / math.sqrt(2)  # (n_pair,512)
        relpos_embds = self.prompter.relpos2embd(relpos_feat)  # (n_pair,512)

        relpos_embds = F.normalize(relpos_embds,dim=-1)
        combined_embds = F.normalize(so_embds+relpos_embds,dim=-1)
        combined_embds = combined_embds[:,:,None] # (n_pair,512,1)

        giou_tags,prompt_ids,counts = self.get_giou_tags(rel_gious,self.giou_th)
        classifier_weights = self.group_classifier_weights[prompt_ids,:,:]  # (n_pair,n_cls,512)
        
        logits = torch.bmm(classifier_weights,combined_embds) / self.temperature 
        # (n_pair,n_cls,512) x (n_pair,512,1) --> (n_pair,n_cls,1)
        logits = logits.squeeze(2)  # (n_pair,n_cls)
     
        pred_probs = torch.sigmoid(logits)     # (n_pair, num_cls) , exclude __background__
        scores,cls_ids = _to_predict_cls_ids(  # (n_pair,k)
            cls_split,
            self.num_base,
            self.num_novel,
            pred_probs,
            pred_topk,
        )
        

        return scores,cls_ids,pair_ids


#### RandomSelect (for ablation)

class AlproPromptTrainer_GroupedRandom(AlproPromptTrainer_Grouped):
    def get_giou_tags(self,rel_gious,giou_th):
        
        n_pair = rel_gious.shape[0]
        random_prompt_ids = torch.randint(0,6,size=(n_pair,),device=rel_gious.device)

        return None,random_prompt_ids,None


#### Single (unified prompt) for ablation


class PromptLearner_Single(nn.Module):

    def __init__(
        self,n_context,alpro_model,cls_split_info_path,use_pos=True
    ):
        super().__init__()

        self.use_pos = use_pos
        cls_split_info = load_json(cls_split_info_path)
        self.num_base = sum([v=="base" for v in cls_split_info["cls2split"].values()])
        self.num_novel = sum([v=="novel" for v in cls_split_info["cls2split"].values()])

        cls2id_map = cls_split_info["cls2id"]
        cls_names = sorted(cls2id_map.items(),key= lambda x:x[1])
        cls_names = [x[0] for x in cls_names]  # including __background__
        cls_names = [name.replace("_", " ") for name in cls_names]
        name_lens = [len(name.split(" ")) for name in cls_names]
        self.n_cls = len(cls_names)
        self.n_ctx = n_context
        self.max_txt_len = GeneralAlproCfg.max_txt_len
        assert all([len_ + self.n_ctx <= self.max_txt_len for len_ in name_lens])

        place_holder_strs = " ".join(["X"] * self.n_ctx)
        token_strs = [place_holder_strs + " " + name for name in cls_names] # including __background__

        tokenizer = BertTokenizerFast.from_pretrained("/home/gkf/project/ALPRO/ext/bert-base-uncased")
        batch_enc = tokenizer.batch_encode_plus(
            token_strs,  # bsz= n_cls
            max_length= self.max_txt_len,  # default: 40
            padding='max_length',
            return_tensors="pt",
            truncation=True
        )
        token_ids = batch_enc.input_ids  # (n_cls, max_L)  # max_L == 40
        token_mask = batch_enc.attention_mask  # (n_cls, max_L)
        # [CLS] t1 t2 ... tn [SEP] 0, 0, 0 ..., 0 (zero-padding to max_L == 40) refer to `tools/token_ids.png`

        with torch.no_grad():
            token_embds =  alpro_model.text_encoder.bert.embeddings.word_embeddings(token_ids)  # (n_cls,40,768), n_cls  including __background__
        
        prefix_embds =  token_embds[:, :1, :]  # (n_cls,1,768), [CLS] token embedding
        suffix_embds =  token_embds[:, 1 + self.n_ctx :, :]  # (n_cls, 40-1-n_ctx ,768) embedding of cls_name tokens, [SEP] token, and zero-pad tokens

        self.register_buffer("prefix_embds", prefix_embds)
        self.register_buffer("suffix_embds", suffix_embds)
        self.register_buffer("token_mask",token_mask)

        self.setup_learnable_parameters()

    def setup_learnable_parameters(self):
        ctx_embds = torch.empty(self.n_ctx, 768)
        nn.init.normal_(ctx_embds, std=0.02)
        self.ctx_embds = nn.Parameter(ctx_embds,requires_grad=True)  # to be optimized

        if self.use_pos:
            self.relpos2embd = nn.Sequential(
                nn.Linear(12,256),
                nn.ReLU(),
                nn.Linear(256,256,bias=False)
            )
    
    def specify_clsids_range(self,split):
        if split == "base":
            pids_list = list(range(1,self.num_base+1))   # (1,2,...,92), len==92,   exclude __background__
        elif split == "novel":
            pids_list = list(range(self.num_base+1,self.num_base+self.num_novel+1))
            # (93,94,...,132), len == 40
        elif split == "all":
            pids_list = list(range(1,self.num_base+self.num_novel+1))    # len==132, i.e., 1 ~ 132
        else:
            assert False, "split must be base, novel, or all"
        
        return pids_list


    def forward(self,split):
        
        pids_list = self.specify_clsids_range(split.lower())

        n_cls = len(pids_list)
        prefix = self.prefix_embds[pids_list,:,:]
        suffix = self.suffix_embds[pids_list,:,:]
        token_mask = self.token_mask[pids_list,:]
        ctx = self.ctx_embds

        ctx = ctx.unsqueeze(0).expand(n_cls, -1, -1)

        ctx_token_embds = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)  # [CLS] token
                ctx, # (n_cls, n_ctx, dim)  # context tokens
                suffix,  # (n_cls, *, dim)  # * refers to cls_name tokens, [SEP] token, and zero-pad tokens
            ],
            dim=1,
        )

       
        return ctx_token_embds,token_mask


class AlproPromptTrainer_Single(nn.Module):

    def __init__(self,configs,is_train=True,train_on_gt_only=False):
        super().__init__()

        self.train_on_gt_only = train_on_gt_only
        self.num_base = configs["num_base"]
        self.num_novel = configs["num_novel"]
        self.temperature = configs["temperature"]  # alpro learned temperature
        n_context = configs["n_context_tokens"]
        pred_cls_split_info_path = configs["pred_cls_split_info_path"]

        alpro_model = setup_alpro_model()
        self.prompter = PromptLearner_Single(n_context,alpro_model,pred_cls_split_info_path)
        self.text_encoder = AlproTextEncoder(alpro_model)


    def state_dict(self):

        return self.prompter.state_dict()
    
    def load_state_dict(self,state_dict):

        self.prompter.load_state_dict(state_dict)
    

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

        so_embds = F.normalize(s_embds - o_embds,dim=-1)

        token_embds,token_mask = self.prompter("base")    # (n_cls,max_L,768)
        classifier_weights = self.text_encoder(token_embds,token_mask)  # (n_cls,256)

        relpos_embds = self.prompter.relpos2embd(relpos_feats)
        relpos_embds = F.normalize(relpos_embds,dim=-1)
        combined_embds = F.normalize(so_embds+relpos_embds,dim=-1)

        logits = torch.matmul(combined_embds,classifier_weights.t()) / self.temperature  # (bsz,n_cls)


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
            "total":loss.detach(),
        }
        return loss,loss_for_show

    
    def forward(self,batch_data,cls_split):
        if self.train_on_gt_only:
            return self.forward_on_gt_only(batch_data)
        # TODO
        return None

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


    def reset_classifier_weights(self,cls_split):
        # this is used in test , reset for each epoch

        # for each epoch， reset once and save the classifier_weights as buffer,
        # reset at each iteration is not necessary and is too time consuming
        # and we must re-reset for each epoch

        token_embds,token_mask = self.prompter(cls_split)    # (num_base,max_L,768)
        classifier_weights = self.text_encoder(token_embds,token_mask)  # (num_base,256)

        self.register_buffer("classifier_weights",classifier_weights,persistent=False)


    def forward_inference_bsz1(self,data,cls_split,pred_topk=10):
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


        so_embds = F.normalize(s_embds - o_embds,dim=-1)
        relpos_embds = self.prompter.relpos2embd(relpos_feat)  # (n_pair,512)

        relpos_embds = F.normalize(relpos_embds,dim=-1)
        combined_embds = F.normalize(so_embds+relpos_embds,dim=-1)

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



if __name__ == "__main__":

    alpro_model = setup_alpro_model()
    xx = PromptLearner(10,alpro_model)

    for name,v in xx.named_parameters():
        print(name,v.shape)
    print(xx.prefix_embds.shape)
    print(xx.suffix_embds.shape)

    for split in ["base","novel","all"]:
        aa = xx.specify_clsids_range(split)
        print(aa)
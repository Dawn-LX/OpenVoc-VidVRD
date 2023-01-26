## this file is copied and modified from /home/gkf/project/ALPRO/src/modeling/alpro_models.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from horovod import torch as hvd
######################## gkf ##################
#  we comment out all the code about horovod (hvd)
# beacuse we only use the model for inference
# and all the code of hvd is used for training & loss & multi-gpu


from .timesformer.vit import TimeSformer
from .xbert import BertForMaskedLM



class AlproBaseModel(nn.Module):
    def __init__(self, config=None, input_format='RGB', video_enc_cfg=None, temp=0.07):
        super().__init__()
        
        self.temp = nn.Parameter(torch.ones([]) * temp)   

        self.bert_config = config

        visual_model_cls = eval(video_enc_cfg['cls'])  # `TimeSformer`

        self.visual_encoder = visual_model_cls(model_cfg=video_enc_cfg, input_format=input_format, cross_attention_config=config)
        # in our setting, visual_encoder is TimeSformer
        self.text_encoder = BertForMaskedLM.from_pretrained('bert-base-uncased', config=self.bert_config)

        # FIXME make them configurable
        embed_dim = 256
        vision_width = 768

        text_width = self.bert_config.hidden_size  # 768

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)         

        self.itc_token_type = self.bert_config.itc_token_type
        self.itm_head = nn.Linear(text_width, 2)     


    def load_separate_ckpt(self, visual_weights_path=None, bert_weights_path=None):
        if visual_weights_path:
            self.visual_encoder.load_state_dict(visual_weights_path)

        # if bert_weights_path:
        #     load_multimodal_encoder_state_dict_with_mismatch(self.cross_encoder, bert_weights_path)
        #     load_mlm_head_state_dict_with_mismatch(self.mlm_head, bert_weights_path)

    # def freeze_cnn_backbone(self):
    #     for n, p in self.visual_encoder.feature.named_parameters():
    #         p.requires_grad = False



class AlproForVideoTextRetrieval(AlproBaseModel):
    """
    """
    def __init__(self, config, video_enc_cfg, input_format='RGB'):
        super(AlproForVideoTextRetrieval, self).__init__(config, input_format=input_format, video_enc_cfg=video_enc_cfg)

    def forward(self, batch):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)  # what the fuck ???

        visual_inputs = batch['visual_inputs']
        text_input_mask = batch['text_input_mask']
        text_input_ids = batch['text_input_ids']

        device = visual_inputs.device

        b, t, c, h, w = visual_inputs.shape
        # timeSformer asks for (b, c, t, h, w) as input.
        # visual embeddings
        visual_inputs = visual_inputs.transpose(1, 2)

        video_embeds = self.visual_encoder.forward_features(visual_inputs, return_all_tokens=True)
        # image_embeds = image_embeds.repeat(text_input_mask.shape[0], 1, 1)
        video_feat = F.normalize(self.vision_proj(video_embeds[:,0,:]),dim=-1)  

        video_atts = torch.ones(video_embeds.size()[:-1],dtype=torch.long).to(device)

        # text embeddings
        text_output = self.text_encoder.bert(text_input_ids,
                                             attention_mask=text_input_mask,
                                             return_dict=True,
                                             mode='text'
                                            )
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)                 

        # ========== (in-batch) ITC loss ==========
        gathered_video_feats = hvd.allgather(video_feat)
        gathered_text_feats = hvd.allgather(text_feat)

        sim_v2t = video_feat @ gathered_text_feats.t() / self.temp 
        sim_t2v = text_feat @ gathered_video_feats.t() / self.temp 

        sim_targets = torch.zeros_like(sim_v2t)

        local_rank = hvd.local_rank()
        b_start, b_end = b * local_rank, b * (local_rank + 1)
        sim_targets[:, b_start: b_end] = torch.eye(b)

        loss_v2t = -torch.sum(F.log_softmax(sim_v2t, dim=1)*sim_targets,dim=1).mean()
        loss_t2v = -torch.sum(F.log_softmax(sim_t2v, dim=1)*sim_targets,dim=1).mean() 

        vtc_loss = (loss_v2t+loss_t2v) / 2

        # ========= ITM ==========
        text_atts = batch['text_input_mask']

        # non-masked text and non-masked image 
        vtm_loss, vtm_logits, vtm_labels = self.compute_vtm(text_embeds=text_embeds, 
                                                            text_atts=text_atts, 
                                                            image_embeds=video_embeds, 
                                                            image_atts=video_atts, 
                                                            sim_i2t=sim_v2t.clone(), # for hard mining
                                                            sim_t2i=sim_t2v.clone()  # for hard mining
                                                           )

        return dict(
            itm_scores=vtm_logits,
            itm_loss=vtm_loss,
            itm_labels=vtm_labels,
            itc_loss=vtc_loss
        )
    
    def compute_vtm(self, text_embeds, text_atts, image_embeds, image_atts, sim_i2t, sim_t2i):
        device = text_embeds.device

        # ====== positive pairs =======
        attention_mask = torch.cat([text_atts, image_atts], dim=1)
        embedding_output_pos = torch.cat([text_embeds, image_embeds], dim=1)

        encoder_outputs_pos = self.text_encoder.bert(encoder_embeds=embedding_output_pos,
                                                     attention_mask=attention_mask,
                                                     return_dict=True,
                                                     mode='fusion'
                                                    )

        # ====== negative pairs =======
        bs = text_embeds.shape[0] 

        local_rank = hvd.local_rank()
        b_start, b_end = bs * local_rank, bs * (local_rank + 1)

        with torch.no_grad():
            weights_v2t = sim_i2t[:,b_start:b_end]
            weights_t2v = sim_t2i[:,b_start:b_end]
   
            # never select self as negative
            weights_v2t.fill_diagonal_(-np.Inf)
            weights_t2v.fill_diagonal_(-np.Inf)

            weights_v2t = F.softmax(weights_v2t, dim=1)
            weights_t2v = F.softmax(weights_t2v, dim=1)

        # select a negative image for each text
        # FIXME to optimize using indexing operations
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2v[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_v2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text_atts[neg_idx])

        text_embeds_neg = torch.stack(text_embeds_neg,dim=0)   
        text_atts_neg = torch.stack(text_atts_neg,dim=0)      

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg],dim=0)     
        text_atts_all = torch.cat([text_atts, text_atts_neg],dim=0)     

        video_embeds_all = torch.cat([image_embeds_neg,image_embeds],dim=0)
        video_atts_all = torch.cat([image_atts,image_atts],dim=0)

        attention_mask_all = torch.cat([text_atts_all, video_atts_all], dim=1)
        embedding_output_all = torch.cat([text_embeds_all, video_embeds_all], dim=1)

        # forward negative pairs via cross encoder
        encoder_outputs_neg = self.text_encoder.bert(encoder_embeds=embedding_output_all,
                                                     attention_mask=attention_mask_all,
                                                     return_dict=True,
                                                     mode='fusion'
                                                    )

        vl_embeddings = torch.cat([encoder_outputs_pos.last_hidden_state[:,0,:], 
                                   encoder_outputs_neg.last_hidden_state[:,0,:]],dim=0)
        vtm_logits = self.itm_head(vl_embeddings)            

        vtm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)], dim=0).to(device)
        vtm_loss = F.cross_entropy(vtm_logits, vtm_labels)     

        return vtm_loss, vtm_logits, vtm_labels 

    def forward_inference(self, batch):
        visual_inputs = batch['visual_inputs']
        text_input_mask = batch['text_input_mask']  # (num_text,max_L)
        text_input_ids = batch['text_input_ids']    # (num_text,max_L)
        # gkf: TODO: figure out the format (shape) of  text_input_mask & text_input_ids
        # refer to `class VideoRetrievalCollator(object):` in `src/datasets/dataset_video_retrieval.py`

        device = visual_inputs.device

        b, t, c, h, w = visual_inputs.shape  # gkf: here c==3 for RGB video
        # timeSformer asks for (b, c, t, h, w) as input.
        visual_inputs = visual_inputs.transpose(1, 2)

        video_embeds = self.visual_encoder.forward_features(visual_inputs, return_all_tokens=True) # (1,197,768)
        video_feat = F.normalize(self.vision_proj(video_embeds[:,0,:]),dim=-1)    # (1,256)

        video_embeds = video_embeds.repeat(text_input_mask.shape[0], 1, 1)  # (1,256) --> (num_text,256)
        # image_feat = image_feat.repeat(text_input_mask.shape[0], 1)

        video_atts = torch.ones(video_embeds.size()[:-1],dtype=torch.long).to(device)
        text_output = self.text_encoder.bert(text_input_ids,
                                             attention_mask=text_input_mask,
                                             return_dict=True,
                                             mode='text'
                                            )
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)                 

        vtc_sim_scores = video_feat @ text_feat.t() / self.temp  # (1,256) @ (256,num_text)  --> (1,num_text)
        #gkf: the @ pytorch means matrix multiplication , this is equivalent to cosine similarity

        attention_mask = torch.cat([text_input_mask, video_atts], dim=1)
        embedding_output = torch.cat([text_embeds, video_embeds], dim=1)

        encoder_outputs = self.text_encoder.bert(encoder_embeds=embedding_output,
                                                 attention_mask=attention_mask,
                                                 return_dict=True,
                                                 mode='fusion'
                                                )

        vl_embeddings = encoder_outputs.last_hidden_state[:,0,:]
        logits = self.itm_head(vl_embeddings)  # 这里只做一个二分类， positive or negative
        # refer to Sec 3.2 of  `Align before Fuse: Vision and Language Representation Learning with Momentum Distillation NIPS2021`

        return dict(logits=logits, itc_scores=vtc_sim_scores)


# ----------------------------- added by gkf -----------------------------
class AlproForVideoFeatExtract(AlproBaseModel):
    def __init__(self, config, video_enc_cfg, input_format='RGB'):
        super().__init__(config, input_format=input_format, video_enc_cfg=video_enc_cfg)
    
    def forward(self,visual_inputs):
        '''We only implement forward for inference'''

        # text_input_mask = batch['text_input_mask']
        # text_input_ids = batch['text_input_ids']
        
        device = visual_inputs.device

        bsz, t, c, h, w = visual_inputs.shape
        # timeSformer asks for (bsz, c, t, h, w) as input.
        visual_inputs = visual_inputs.transpose(1, 2)

        # self.visual_encoder.model.cls_token += soft_prompt  # shape == (1,1,768)

        video_embeds = self.visual_encoder.forward_features(visual_inputs, return_all_tokens=True)  # (bsz,197,768)
        video_feats = self.vision_proj(video_embeds[:,0,:])   # (bsz,256), in our setting bsz is num_traj
        # i.e., use the first token [CLS] token to represent the sequence
        # print(video_embeds.shape,video_feats.shape)
        return video_feats


class AlproForTextFeatExtract(AlproBaseModel):
    def __init__(self, config, tokenizer,max_txt_len,video_enc_cfg, input_format='RGB'):
        super().__init__(config, input_format=input_format, video_enc_cfg=video_enc_cfg)
        # TODO avoid video_enc_cfg input

        self.tokenizer = tokenizer
        self.max_length = max_txt_len  # default:40
        
    
    def forward(self,text_str_list):
        '''We only implement forward for inference'''

        # gkf: NOTE the format (shape) of  text_input_mask & text_input_ids
        # refer to `class VideoRetrievalCollator(object):` in `src/datasets/dataset_video_retrieval.py`
        device = next(self.parameters()).device

        B = len(text_str_list)
        batch_enc = self.tokenizer.batch_encode_plus(
            text_str_list,
            max_length=self.max_length,
            padding='max_length',
            return_tensors="pt",
            truncation=True
        )
        text_input_ids = batch_enc.input_ids.to(device)  # (B, L)  # include [CLS] & [SEP] tokens
        # e.g., [CLS] t1 t2 ... tn [SEP] 0, 0, 0 ..., 0 (zero-padding to max_length == 40) refer to `tools/token_ids.png`
        text_input_mask = batch_enc.attention_mask.to(device)  # (B, L)
        # num_tokens = text_input_mask.sum(dim=-1)
        # print(num_tokens)
        text_output = self.text_encoder.bert(text_input_ids,
                                             attention_mask=text_input_mask,
                                             return_dict=True,
                                             mode='text'
                                            )
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)                 

        return text_feat


class AlproTextWithTokenEmbd(AlproBaseModel):
    def __init__(self, config, tokenizer,max_txt_len,video_enc_cfg, input_format='RGB'):
        super().__init__(config, input_format=input_format, video_enc_cfg=video_enc_cfg)
        # TODO avoid video_enc_cfg input

        self.tokenizer = None
        self.max_length = max_txt_len  # default:40
        
    
    def forward(self,token_embds,token_mask):
        '''We only implement forward for inference'''
        # token_embds.shape == (bsz, max_L, dim_emb) == (n_str+2, 768), 2 stands for [CLS] & [SEP] tokens
        # NOTE: note that the padded embds are not all-zero, we use the learned embds (indexed by 0) in Alpro-text's bert
        # token_mask.sahpe == (bsz,max_L)
        bsz,max_L,_ = token_embds.shape
        assert max_L == self.max_length


        text_output = self.text_encoder.bert(inputs_embeds=token_embds,
                                             attention_mask=token_mask,
                                             return_dict=True,
                                             mode='text'
                                            )
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)                 

        return text_feat
  
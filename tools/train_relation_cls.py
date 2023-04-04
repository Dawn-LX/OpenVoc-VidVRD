
import root_path
import argparse
import os
from tqdm import tqdm
from collections import defaultdict

import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from models.TrajClsModel_v2 import OpenVocTrajCls as OpenVocTrajCls_NoBgEmb

from models.PromptModels_v3 import AlproPromptTrainer,AlproPromptTrainer_Grouped,AlproPromptTrainer_GroupedRandom,AlproPromptTrainer_Single
from models.RelationClsModel_v3 import OpenVocRelCls_FixedPrompt,OpenVocRelCls_LearnablePrompt,OpenVocRelCls_stage2,OpenVocRelCls_stage2_Grouped,OpenVocRelCls_stage2_GroupedRandom,OpenVocRelCls_stage2_Single,VidVRDII_FixedPrompt

from dataloaders.dataset_vidvrd_v2 import VidVRDGTDatasetForTrain,VidVRDUnifiedDataset,VidVRDGTDatasetForTrain_GIoU,VidVRDUnifiedDataset_GIoU
from utils.config_parser import parse_config_py
from utils.evaluate import EvalFmtCvtor
from utils.utils_func import get_to_device_func,load_json
from utils.logger import LOGGER, add_log_to_file
from VidVRDhelperEvalAPIs import eval_visual_relation,evaluate_v2


def load_checkpoint(model,optimizer,scheduler,ckpt_path):
    checkpoint = torch.load(ckpt_path,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    scheduler.load_state_dict(checkpoint['sched_state_dict'])
    crt_epoch = checkpoint["crt_epoch"]
    batch_size = checkpoint["batch_size"]

    return model,optimizer,scheduler,crt_epoch,batch_size

def save_checkpoint(batch_size,crt_epoch,model,optimizer,scheduler,save_path):
    checkpoint = {
        "batch_size":batch_size,
        "crt_epoch":crt_epoch + 1,
        "model_state_dict":model.state_dict(),
        "optim_state_dict":optimizer.state_dict(),
        "sched_state_dict":scheduler.state_dict(),
    }
    torch.save(checkpoint,save_path)

def modify_state_dict(state_dict):
    # NOTE This function is temporary
    
    text_embeddings = state_dict.pop("text_embeddings") # (36,dim_emb)
    background_emb = state_dict.pop("background_emb")  # (dim_emb,)
    class_embeddings = torch.cat((background_emb[None,:],text_embeddings[1:,:]),dim=0)

    state_dict.update({"class_embeddings":class_embeddings})

    return state_dict


class SegmentEvaluater(object):
    def __init__(self,
        eval_dataset_cfg,
        model_traj_cfg,
        eval_cfg,
        device,
        enti_split_info_path = "configs/VidVRD_class_spilt_info.json",
        pred_split_info_path = "configs/VidVRD_pred_class_spilt_info_v2.json",
        eval_split_traj = "all",
        eval_split_pred = "novel",
        eval_dataset_class = VidVRDUnifiedDataset,
    ):

        LOGGER.info("preparing eval dataloader ...")
        eval_dataset = eval_dataset_class(**eval_dataset_cfg)
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=1,
            collate_fn = lambda x :x[0],
            num_workers = 2,
            drop_last= False,
            shuffle= False,
        )
        LOGGER.info(" --------------- eval dataloader ready. ---------------")

        model_traj = OpenVocTrajCls_NoBgEmb(model_traj_cfg,is_train=False)
        LOGGER.info("loading check point from {}".format(eval_cfg["ckpt_path_traj"]))
        check_point = torch.load(eval_cfg["ckpt_path_traj"],map_location=torch.device('cpu'))
        state_dict = check_point["model_state_dict"]
        model_traj = model_traj.to(device)
        model_traj.load_state_dict(state_dict)
        model_traj.eval()
        model_traj.reset_classifier_weights(eval_split_traj)

        convertor = EvalFmtCvtor(
            "vidvrd",
            enti_split_info_path,
            pred_split_info_path,
            score_merge="mean",
            segment_cvt=True
        )



        traj_cls_info = load_json(enti_split_info_path)
        pred_cls_info = load_json(pred_split_info_path)
        traj_categories = [c for c,s in traj_cls_info["cls2split"].items() if (s == eval_split_traj) or eval_split_traj=="all"]
        traj_categories = set([c for c in traj_categories if c != "__background__"])
        pred_categories = [c for c,s in pred_cls_info["cls2split"].items() if (s == eval_split_pred) or eval_split_pred=="all"]
        pred_categories = set([c for c in pred_categories if c != "__background__"])

        gt_relations = load_json("datasets/gt_jsons/VidVRDtest_segment_gts.json")
        gt_relations_ = defaultdict(list)
        for vsig,relations in gt_relations.items(): # same format as prediction results json, refer to `VidVRDhelperEvalAPIs/README.md`
            for rel in relations:
                s,p,o = rel["triplet"]
                if not ((s in traj_categories) and (p in pred_categories) and (o in traj_categories)):
                    continue
                gt_relations_[vsig].append(rel)
        
        self.dataloader = eval_dataloader
        self.model_traj = model_traj
        self.convertor = convertor
        self.gt_relations = gt_relations_
        self.eval_split_traj = eval_split_traj
        self.eval_split_pred = eval_split_pred
        self.eval_cfg = eval_cfg
        self.device = device
        self.to_device_func = get_to_device_func(device)



    def evaluate(self,model_pred):
        
        pred_topk = self.eval_cfg["pred_topk"]
        return_triplets_topk = self.eval_cfg["return_triplets_topk"]
        if hasattr(model_pred,"reset_classifier_weights"):
            with torch.no_grad():
                model_pred.reset_classifier_weights(self.eval_split_pred)  # default: "novel"
        
        prediction_results = dict()
        for data in tqdm(self.dataloader):
            # for simplicity, we set batch_size = 1 at inference time
            # seg_tag,det_traj_info,traj_pair_info,rel_pos_feat,labels

            (
                seg_tag,
                det_traj_info,
                traj_embds,
                traj_ids_aft_filter,
                rel_pos_feat,  # relative pos_feate, or tuple(rel_pos_feat,rel_giou)
                labels
            ) = data

            det_feats  = det_traj_info["features"]
            traj_bboxes = det_traj_info["bboxes"]
            traj_starts = det_traj_info["fstarts"]
            n_det = det_feats.shape[0]

            input_data = (
                det_feats,
                traj_embds,
                rel_pos_feat
            )
            
            input_data = tuple(self.to_device_func(x) for x in input_data)
            
            with torch.no_grad():
                traj_scores,traj_cls_ids = self.model_traj.forward_inference_bsz1(input_data[0])  # (n_det,) # before filter
                if hasattr(model_pred,"conditioned_on_enti_cls") and model_pred.conditioned_on_enti_cls:
                    input_data = input_data + (traj_cls_ids,)
                p_scores,p_clsids,pair_ids = model_pred.forward_inference_bsz1(input_data,self.eval_split_pred,pred_topk) # (n_pair,k)

            s_ids = pair_ids[:,0]  # (n_pair,)
            o_ids = pair_ids[:,1] 

            s_clsids = traj_cls_ids[s_ids]  # (n_pair,)
            o_clsids = traj_cls_ids[o_ids]
            s_scores = traj_scores[s_ids]
            o_scores = traj_scores[o_ids]

            n_pair,k = p_clsids.shape
            triplet_scores = torch.stack([
                p_scores.reshape(-1),
                s_scores[:,None].repeat(1,k).reshape(-1),
                o_scores[:,None].repeat(1,k).reshape(-1)
            ],dim=-1) # (n_pair*k,3)
            triplet_5tuple = torch.stack([
                p_clsids.reshape(-1),
                s_clsids[:,None].repeat(1,k).reshape(-1),
                o_clsids[:,None].repeat(1,k).reshape(-1),
                s_ids[:,None].repeat(1,k).reshape(-1),
                o_ids[:,None].repeat(1,k).reshape(-1),
            ],dim=-1) #  shape == (n_pair*k,5)  # format: [pred_catid,subj_catid,obj_catid,subj_tid,obj_tid]


            traj_bboxes = [tb.cpu() for tb in traj_bboxes]
            traj_starts = traj_starts.cpu()
            triplet_scores = triplet_scores.cpu()
            triplet_5tuple = triplet_5tuple.cpu()
            
            result_per_seg = self.convertor.to_eval_json_format(
                seg_tag,
                triplet_5tuple,
                triplet_scores,
                traj_bboxes,
                traj_starts,
                triplets_topk=return_triplets_topk,
            )
            prediction_results.update(result_per_seg)

        mean_ap, rec_at_n, mprec_at_n,hit_infos = evaluate_v2(self.gt_relations,prediction_results,viou_threshold=0.5)

        metrics = {
            "mAP":mean_ap,
            "R@50":rec_at_n[50],
            "R@100":rec_at_n[100],
            "P@1":mprec_at_n[1],
            "P@5":mprec_at_n[5],
            "P@10":mprec_at_n[10],
        }
        return metrics



def train(model_class,train_dataset_class,eval_dataset_class,args):
    '''
    This func uses `VidVRDUnifiedDataset`
    '''
    cfg_path = args.cfg_path
    output_dir = args.output_dir
    from_checkpoint = args.from_checkpoint
    ckpt_path = args.ckpt_path
    save_tag = args.save_tag

    if output_dir is None:
        output_dir = os.path.dirname(cfg_path)
    log_dir = os.path.join(output_dir,'logfile/')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tensorboard_log_dir = os.path.join(log_dir,"tensorboard_{}/".format(save_tag))
    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)
    if not from_checkpoint:
        os.system("rm {}events*".format(tensorboard_log_dir))
    writer = SummaryWriter(tensorboard_log_dir)

    log_path = os.path.join(log_dir,f'train_{save_tag}.log')
    add_log_to_file(log_path)

    configs = parse_config_py(cfg_path)
    train_dataset_cfg = configs["train_dataset_cfg"]
    eval_dataset_cfg = configs["eval_dataset_cfg"]
    model_cfg = configs["model_pred_cfg"]
    model_traj_cfg = configs["model_traj_cfg"]
    train_cfg = configs["train_cfg"]
    eval_cfg = configs["eval_cfg_for_train"]
    device = torch.device("cuda")
    # device = torch.device("cpu")
    to_device_func = get_to_device_func(device)


    LOGGER.info("train dataset config: {}".format(train_dataset_cfg))
    LOGGER.info("eval dataset config: {}".format(eval_dataset_cfg))
    LOGGER.info("model config: {}".format(model_cfg))
    LOGGER.info("training config: {}".format(train_cfg))

    batch_size          = train_cfg["batch_size"]
    total_epoch         = train_cfg["total_epoch"]
    initial_lr          = train_cfg["initial_lr"]
    lr_decay            = train_cfg["lr_decay"]
    epoch_lr_milestones = train_cfg["epoch_lr_milestones"]


    model = model_class(model_cfg,is_train=True)
    model = model.to(device)
    if hasattr(model,"reset_classifier_weights"):
        with torch.no_grad():
            model.reset_classifier_weights("base")  # this is deprecated
    

    evaluater = SegmentEvaluater(
        eval_dataset_cfg,
        model_traj_cfg,
        eval_cfg,
        device,
        enti_split_info_path = "configs/VidVRD_class_spilt_info.json",
        pred_split_info_path = "configs/VidVRD_pred_class_spilt_info_v2.json",
        eval_split_traj = args.eval_split_traj,
        eval_split_pred = args.eval_split_pred,
        eval_dataset_class=eval_dataset_class
    )
    

    LOGGER.info("preparing train dataloader...")
    train_dataset = train_dataset_class(**train_dataset_cfg)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn = train_dataset.get_collator_func(),
        num_workers = 8,
        drop_last= False,
        shuffle= True,
    )
    LOGGER.info("train dataloader ready.")
    dataset_len = len(train_dataset)
    dataloader_len = len(train_dataloader)
    LOGGER.info(
        "len(dataset)=={},batch_size=={},len(dataloader)=={},{}x{}={}".format(
            dataset_len,batch_size,dataloader_len,batch_size,dataloader_len,batch_size*dataloader_len
        )
    )

    milestones = [int(m*dataset_len/batch_size) for m in epoch_lr_milestones]
    optimizer = torch.optim.Adam(model.parameters(), lr = initial_lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones,gamma=lr_decay)



    if from_checkpoint:
        model,optimizer,scheduler,crt_epoch,batch_size_ = load_checkpoint(model,optimizer,scheduler,ckpt_path)
        # assert batch_size == batch_size_ , "batch_size from checkpoint not match : {} != {}"
        if batch_size != batch_size_:
            LOGGER.warning(
                "!!!Warning!!! batch_size from checkpoint not match : {} != {}".format(batch_size,batch_size_)
            )
        LOGGER.info("checkpoint load from {}".format(ckpt_path))
    else:
        crt_epoch = 0

    LOGGER.info("start training:")
    LOGGER.info("use config: {}".format(cfg_path))
    LOGGER.info("use device: {}".format(device))
    LOGGER.info("weights will be saved in output_dir = {}".format(output_dir))


    it=0
    max_mAP = -1.0
    for epoch in tqdm(range(total_epoch)):
        if epoch < crt_epoch:
            it+=dataloader_len
            continue
        
        model.train()
        epoch_loss = defaultdict(list)
        for batch_data in train_dataloader:
            # seg_tag,det_traj_info,traj_embds,traj_ids_aft_filter,rel_pos_feat,labels
            (
                segment_tags,
                batch_det_traj_info,
                batch_traj_embds,
                batch_traj_ids_aft_filter,
                batch_rel_pos_feat,
                batch_labels
            ) = batch_data
            batch_det_feats  = [det_traj_info["features"] for det_traj_info in batch_det_traj_info]
            batch_data = (
                batch_det_feats,
                batch_traj_embds,
                batch_rel_pos_feat,
                batch_labels
            )

            batch_data = tuple(to_device_func(data) for data in batch_data)

            optimizer.zero_grad()
            total_loss, loss_dict = model(batch_data,cls_split="cls_split is deprecated")
            # TODO average results from muti-gpus
            # combined_loss = combined_loss.mean()
            # loss_dict = {k:v.mean() for k,v in each_loss_term.items()}
            
            total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()
            scheduler.step()

            loss_str = "epoch={};iter={}; ".format(epoch,it)
            for k,v in loss_dict.items():
                epoch_loss[k].append(v.item())
                loss_str += "{}:{:.4f}; ".format(k,v.item())
                writer.add_scalar('Iter/{}'.format(k), v.item(), it)
            loss_str += "lr={}".format(optimizer.param_groups[0]["lr"])
            if it % 10 == 0:
                LOGGER.info(loss_str)
            it+=1


        epoch_loss_str = "mean_loss_epoch={}: ".format(epoch)
        for k,v in epoch_loss.items():
            v = np.mean(v)
            writer.add_scalar('Epoch/{}'.format(k), v, epoch)
            epoch_loss_str += "{}:{:.4f}; ".format(k,v)
        LOGGER.info(epoch_loss_str)

        model.eval()
        LOGGER.info("segment evaluate for epoch-{} ...".format(epoch))
        metrics = evaluater.evaluate(model)
        LOGGER.info("segment eval results of epoch={}: {}".format(epoch,metrics))
        for k,v in metrics.items():
            writer.add_scalar('Epoch_metrics/{}'.format(k), v, epoch)

        if max_mAP < metrics["mAP"]:
            save_path = os.path.join(output_dir,'model_{}_best_mAP.pth'.format(save_tag,epoch))
            save_checkpoint(batch_size,epoch,model,optimizer,scheduler,save_path)
            LOGGER.info("best mAP checkpoint is saved: {} epoch-{}".format(save_path,epoch))
            max_mAP = metrics["mAP"]


        if epoch >0 and epoch % 10 == 0:
            save_path = os.path.join(output_dir,'model_{}_epoch_{}.pth'.format(save_tag,epoch))
            save_checkpoint(batch_size,epoch,model,optimizer,scheduler,save_path)
            LOGGER.info("checkpoint is saved: {}".format(save_path))

    save_path = os.path.join(output_dir,'model_{}_epoch_{}.pth'.format(save_tag,total_epoch))
    save_checkpoint(batch_size,epoch,model,optimizer,scheduler,save_path)
    LOGGER.info("checkpoint is saved: {}".format(save_path))
    LOGGER.handlers.clear()


def train_use_only_gt_data(model_class,train_dataset_class,eval_dataset_class,args):
    '''
    This func uses `VidVRDGTDatasetForTrain`
    '''

    cfg_path = args.cfg_path
    output_dir = args.output_dir
    from_checkpoint = args.from_checkpoint
    ckpt_path = args.ckpt_path
    save_tag = args.save_tag
    
    if output_dir is None:
        output_dir = os.path.dirname(cfg_path)
    log_dir = os.path.join(output_dir,'logfile/')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tensorboard_log_dir = os.path.join(log_dir,"tensorboard_{}/".format(save_tag))
    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)
    if not from_checkpoint:
        os.system("rm {}events*".format(tensorboard_log_dir))
    writer = SummaryWriter(tensorboard_log_dir)

    log_path = os.path.join(log_dir,f'train_{save_tag}.log')
    add_log_to_file(log_path)

    configs = parse_config_py(cfg_path)
    train_dataset_cfg = configs["train_dataset_cfg"]
    eval_dataset_cfg = configs["eval_dataset_cfg"]
    model_cfg = configs["model_pred_cfg"]
    model_traj_cfg = configs["model_traj_cfg"]
    train_cfg = configs["train_cfg"]
    eval_cfg = configs["eval_cfg_for_train"]
    
    device = torch.device("cuda")
    loss_interval = 20
    # device = torch.device("cpu")
    # loss_interval = 2

    to_device_func = get_to_device_func(device)


    LOGGER.info("train dataset config: {}".format(train_dataset_cfg))
    LOGGER.info("eval dataset config: {}".format(eval_dataset_cfg))
    LOGGER.info("model config: {}".format(model_cfg))
    LOGGER.info("training config: {}".format(train_cfg))

    batch_size          = train_cfg["batch_size"]
    total_epoch         = train_cfg["total_epoch"]
    initial_lr          = train_cfg["initial_lr"]
    lr_decay            = train_cfg["lr_decay"]
    epoch_lr_milestones = train_cfg["epoch_lr_milestones"]


    model = model_class(model_cfg,is_train=True,train_on_gt_only=True)
    model = model.to(device)
    if hasattr(model,"reset_classifier_weights"):
        model.reset_classifier_weights("base")  # this is deprecated
        assert model.training

    evaluater = SegmentEvaluater(
        eval_dataset_cfg,
        model_traj_cfg,
        eval_cfg,
        device,
        enti_split_info_path = "configs/VidVRD_class_spilt_info.json",
        pred_split_info_path = "configs/VidVRD_pred_class_spilt_info_v2.json",
        eval_split_traj = args.eval_split_traj,
        eval_split_pred = args.eval_split_pred,
        eval_dataset_class=eval_dataset_class
    )
    

    LOGGER.info("preparing train dataloader...")
    train_dataset = train_dataset_class(**train_dataset_cfg)
    assert train_dataset.bsz_wrt_pair
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn = train_dataset.get_collator_func(),
        num_workers = 8,
        drop_last= False,
        shuffle= True,
    )
    LOGGER.info("train dataloader ready.")
    dataloader_len = len(train_dataloader)
    LOGGER.info(
        "batch_size=={},len(dataloader)=={}".format(batch_size,dataloader_len)
    )

    milestones = [int(m*dataloader_len) for m in epoch_lr_milestones]
    optimizer = torch.optim.Adam(model.parameters(), lr = initial_lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones,gamma=lr_decay)


    if from_checkpoint:
        model,optimizer,scheduler,crt_epoch,batch_size_ = load_checkpoint(model,optimizer,scheduler,ckpt_path)
        # assert batch_size == batch_size_ , "batch_size from checkpoint not match : {} != {}"
        if batch_size != batch_size_:
            LOGGER.warning(
                "!!!Warning!!! batch_size from checkpoint not match : {} != {}".format(batch_size,batch_size_)
            )
        LOGGER.info("checkpoint load from {}".format(ckpt_path))
    else:
        crt_epoch = 0

    LOGGER.info("start training:")
    LOGGER.info("use config: {}".format(cfg_path))
    LOGGER.info("use device: {}".format(device))
    LOGGER.info("weights will be saved in output_dir = {}".format(output_dir))


    it=0
    max_mAP = -1.0
    for epoch in tqdm(range(total_epoch)):
        if epoch < crt_epoch:
            it+=dataloader_len
            continue
        
        model.train()
        epoch_loss = defaultdict(list)

        for batch_data in train_dataloader:
            # seg_tag,det_traj_info,traj_embds,traj_ids_aft_filter,rel_pos_feat,labels
            # (
            #     s_roi_feats,      # (bsz,2048)
            #     o_roi_feats,
            #     s_embds,          # (bsz,256)
            #     o_embds,
            #     relpos_feats,     # (bsz,12)
            #     triplet_cls_ids   # list[tensor] each shape == (n_preds,3)
            # ) = batch_data

            # print(batch_data[2].shape)
            batch_data = tuple(to_device_func(data) for data in batch_data)
            # batch_data = batch_data[:2] + tuple(to_device_func(data) for data in batch_data[2:])
        
            optimizer.zero_grad()
            total_loss, loss_dict = model(batch_data,cls_split="cls_split is deprecated")
            # TODO average results from muti-gpus
            # combined_loss = combined_loss.mean()
            # loss_dict = {k:v.mean() for k,v in each_loss_term.items()}
            
            total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()
            scheduler.step()

            loss_str = "epoch={};iter={}; ".format(epoch,it)
            for k,v in loss_dict.items():
                epoch_loss[k].append(v.item())
                loss_str += "{}:{:.4f}; ".format(k,v.item())
                writer.add_scalar('Iter/{}'.format(k), v.item(), it)
            loss_str += "lr={}".format(optimizer.param_groups[0]["lr"])
            if it % loss_interval == 0:
                LOGGER.info(loss_str)
            it+=1


        epoch_loss_str = "mean_loss_epoch={}: ".format(epoch)
        for k,v in epoch_loss.items():
            v = np.mean(v)
            writer.add_scalar('Epoch/{}'.format(k), v, epoch)
            epoch_loss_str += "{}:{:.4f}; ".format(k,v)
        LOGGER.info(epoch_loss_str)

        model.eval()
        LOGGER.info("segment evaluate for epoch-{} ...".format(epoch))
        metrics = evaluater.evaluate(model)
        LOGGER.info("segment eval results of epoch={}: {}".format(epoch,metrics))
        for k,v in metrics.items():
            writer.add_scalar('Epoch_metrics/{}'.format(k), v, epoch)

        if max_mAP < metrics["mAP"]:
            save_path = os.path.join(output_dir,'model_{}_best_mAP.pth'.format(save_tag,epoch))
            save_checkpoint(batch_size,epoch,model,optimizer,scheduler,save_path)
            LOGGER.info("best mAP checkpoint is saved: {} epoch-{}".format(save_path,epoch))
            max_mAP = metrics["mAP"]


        if epoch >0 and epoch % 10 == 0:
            save_path = os.path.join(output_dir,'model_{}_epoch_{}.pth'.format(save_tag,epoch))
            save_checkpoint(batch_size,epoch,model,optimizer,scheduler,save_path)
            LOGGER.info("checkpoint is saved: {}".format(save_path))

    save_path = os.path.join(output_dir,'model_{}_epoch_{}.pth'.format(save_tag,total_epoch))
    save_checkpoint(batch_size,epoch,model,optimizer,scheduler,save_path)
    LOGGER.info("checkpoint is saved: {}".format(save_path))
    LOGGER.handlers.clear()



if __name__ == "__main__":
    random.seed(111)
    np.random.seed(111)
    torch.random.manual_seed(111)

    parser = argparse.ArgumentParser(description="Object Detection Demo")

    parser.add_argument("--cfg_path", type=str,help="...")
    parser.add_argument("--output_dir", type=str, help="default `output_dir` will be set as the dir of `cfg_path`")
    parser.add_argument("--from_checkpoint", action="store_true")  #
    parser.add_argument("--ckpt_path", type=str,help="...")
    parser.add_argument("--model_class", type=str,default="")
    parser.add_argument("--train_dataset_class", type=str,default="VidVRDGTDatasetForTrain")
    parser.add_argument("--eval_dataset_class", type=str,default="VidVRDUnifiedDataset")
    parser.add_argument("--eval_split_traj", type=str,default="all")
    parser.add_argument("--eval_split_pred", type=str,default="novel")
    
    parser.add_argument("--save_tag", type=str,default="")
    parser.add_argument("--loss_print_interval", type=int,default=20)

    parser.add_argument("--use_gt_only_data", action="store_true")
    # parser.add_argument("--other_cfgs",nargs="+",default=[])

    args = parser.parse_args()

    # model_class = BaselineRelationCls if args.use_baseline else OpenVocRelationCls
    model_class = eval(args.model_class)
    train_dataset_class = eval(args.train_dataset_class)
    eval_dataset_class = eval(args.eval_dataset_class)


    
    if args.use_gt_only_data: # for stage-1 training
        train_use_only_gt_data(
            model_class,
            train_dataset_class,
            eval_dataset_class,
            args
        )
    else:  # for unified training & stage-2 training
        train(
            model_class,
            train_dataset_class,
            eval_dataset_class,
            args
        )

    '''
    ################################ Table-2 ######################################
    ### Table-2 (RePro with both base and novel training data) (RePro_both_BaseNovel_training)
        # stage-1  (A-100 24G memory, 50 epochs total 3.5 hour)
        #-------------------------------- `train_use_only_gt_data` This has been checked 
        TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 python tools/train_relation_cls.py \
            --use_gt_only_data \
            --model_class AlproPromptTrainer_Grouped \
            --train_dataset_class VidVRDGTDatasetForTrain_GIoU \
            --eval_dataset_class VidVRDUnifiedDataset_GIoU \
            --cfg_path  experiments/RelationCls_VidVRD/RePro_both_BaseNovel_training/stage1/cfg_.py \
            --output_dir experiments/RelationCls_VidVRD/RePro_both_BaseNovel_training/stage1/ \
            --eval_split_traj all \
            --eval_split_pred all \
            --save_tag bsz32
        
        # stage-2  (A-100 15G, (about 14791 M), 50 epochs total 2.5 hour )  
        #---------------- `train`
        TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 python tools/train_relation_cls.py \
            --model_class OpenVocRelCls_stage2_Grouped \
            --train_dataset_class VidVRDUnifiedDataset_GIoU \
            --eval_dataset_class VidVRDUnifiedDataset_GIoU \
            --cfg_path experiments/RelationCls_VidVRD/RePro_both_BaseNovel_training/stage2/cfg_.py \
            --output_dir experiments/RelationCls_VidVRD/RePro_both_BaseNovel_training/stage2/ \
            --save_tag bsz32

   
    ################################ Table-3 ######################################

    ### Table-3 (ALPro) AlproVisual_with_FixedPrompt (vanilla_ALPro_inference_only)
        refer to dir:`experiments/RelationCls_VidVRD/vanilla_ALPro_inference_only`
    
        
    ### Table-3 (VidVRDII) VidVRDII_FixedPrompt
        # 2080 Ti; 4.5G memory; 50 epochs total 2.5 hours
        TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 python tools/train_relation_cls.py \
            --model_class VidVRDII_FixedPrompt \
            --train_dataset_class VidVRDUnifiedDataset \
            --eval_dataset_class VidVRDUnifiedDataset \
            --cfg_path experiments/RelationCls_VidVRD/VidVRD_II/cfg_fixedSinglePrompt.py \
            --output_dir experiments/RelationCls_VidVRD/VidVRD_II \
            --save_tag bsz32

        
    ###  Table-3 (RePro*) OpenVocRelCls_LearnablePrompt (RePro_unified_training)
        # 2082 Ti; 7.2G memory; 2.5 hours
        TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 python tools/train_relation_cls.py \
            --model_class OpenVocRelCls_LearnablePrompt \
            --train_dataset_class VidVRDUnifiedDataset \
            --eval_dataset_class VidVRDUnifiedDataset \
            --cfg_path experiments/RelationCls_VidVRD/RePro_unified_training/cfg_.py \
            --output_dir experiments/RelationCls_VidVRD/RePro_unified_training/ \
            --save_tag bsz32
    
    ###  Table-3 (RePro) OpenVocRelCls_stage2_Grouped
    
        # stage-1 2080Ti; 10G (bsz=16); 50 epochs total 6.5 hours
        TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 python tools/train_relation_cls.py \
            --use_gt_only_data \
            --model_class AlproPromptTrainer_Grouped \
            --train_dataset_class VidVRDGTDatasetForTrain_GIoU \
            --eval_dataset_class VidVRDUnifiedDataset_GIoU \
            --cfg_path experiments/RelationCls_VidVRD/RePro/stage1/cfg_.py \
            --output_dir experiments/RelationCls_VidVRD/RePro/stage1 \
            --save_tag bsz16
        
        
        # stage-2 2080 Ti; 10G  50 epoch total 2.5 hours
        TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 python tools/train_relation_cls.py \
            --model_class OpenVocRelCls_stage2_Grouped \
            --train_dataset_class VidVRDUnifiedDataset_GIoU \
            --eval_dataset_class VidVRDUnifiedDataset_GIoU \
            --cfg_path experiments/RelationCls_VidVRD/RePro/stage2/cfg_.py \
            --output_dir experiments/RelationCls_VidVRD/RePro/stage2 \
            --save_tag bsz32

    ################################ Table-4 ######################################

    ###  Table-4 (#1 w/o C, w/o M) OpenVocRelCls_stage2_Single    (ablation_1)

        # stage-1 2080Ti 3.5G; 50 epochs total 2 hours
        TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 python tools/train_relation_cls.py \
            --use_gt_only_data \
            --model_class AlproPromptTrainer_Single \
            --train_dataset_class VidVRDGTDatasetForTrain \
            --eval_dataset_class VidVRDUnifiedDataset \
            --cfg_path   experiments/RelationCls_VidVRD/ablation_1/stage1/cfg_.py \
            --output_dir  experiments/RelationCls_VidVRD/ablation_1/stage1 \
            --save_tag bsz32

        # stage-2 2080Ti 6.0G; 50 epochs total 2.5 hours
        TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 python tools/train_relation_cls.py \
            --model_class OpenVocRelCls_stage2_Single \
            --train_dataset_class VidVRDUnifiedDataset \
            --eval_dataset_class VidVRDUnifiedDataset \
            --cfg_path  experiments/RelationCls_VidVRD/ablation_1/stage2/cfg_.py \
            --output_dir experiments/RelationCls_VidVRD/ablation_1/stage2 \
            --save_tag bsz32

    ###  Table-4 (#2 w C, w/o M) OpenVocRelCls_stage2    (ablation_2)

        # stage-1 2080Ti 4.5G; 50 epochs total 2.0 hours
        TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 python tools/train_relation_cls.py \
            --use_gt_only_data \
            --model_class AlproPromptTrainer \
            --train_dataset_class VidVRDGTDatasetForTrain \
            --eval_dataset_class VidVRDUnifiedDataset \
            --cfg_path experiments/RelationCls_VidVRD/ablation_2/stage1/cfg_.py \
            --output_dir experiments/RelationCls_VidVRD/ablation_2/stage1 \
            --save_tag bsz32
        
        # stage-2 2080Ti 6.6G; 50 epochs total 2.5 hours
        TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 python tools/train_relation_cls.py \
            --model_class OpenVocRelCls_stage2 \
            --train_dataset_class VidVRDUnifiedDataset \
            --eval_dataset_class VidVRDUnifiedDataset \
            --cfg_path experiments/RelationCls_VidVRD/ablation_2/stage2/cfg_.py \
            --output_dir experiments/RelationCls_VidVRD/ablation_2/stage2 \
            --save_tag bsz32

    ###  Table-4 (#3 Ens, #4 Rand)   OpenVocRelCls_stage2_MeanEnsemble (ablation_3_4)
        NOTE: #3 Ens & #4 Rand only differs at testing time (refer to Section4.5 of the paper)

        # stage-1 2080Ti; 10G (bsz=16); 50 epochs total about 6.5 hours
        TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 python tools/train_relation_cls.py \
            --use_gt_only_data \
            --model_class AlproPromptTrainer_GroupedRandom \
            --train_dataset_class VidVRDGTDatasetForTrain_GIoU \
            --eval_dataset_class VidVRDUnifiedDataset_GIoU \
            --cfg_path experiments/RelationCls_VidVRD/ablation_3_4/stage1/cfg_.py \
            --output_dir experiments/RelationCls_VidVRD/ablation_3_4/stage1 \
            --save_tag bsz32

        # stage-2 2080Ti; 10G; 50 epochs total 2 hours 40 min
        TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 python tools/train_relation_cls.py \
            --model_class OpenVocRelCls_stage2_GroupedRandom \
            --train_dataset_class VidVRDUnifiedDataset_GIoU \
            --eval_dataset_class VidVRDUnifiedDataset_GIoU \
            --cfg_path experiments/RelationCls_VidVRD/ablation_3_4/stage2/cfg_.py \
            --output_dir experiments/RelationCls_VidVRD/ablation_3_4/stage2 \
            --save_tag bsz32

#### stage-1
tensorboard --logdir_spec=\
ablation_1:/home/gaokaifeng/project/OpenVoc-VidVRD/experiments/RelationCls_VidVRD/ablation_1/stage1/logfile/tensorboard_bsz32,\
ablation_2:/home/gaokaifeng/project/OpenVoc-VidVRD/experiments/RelationCls_VidVRD/ablation_2/stage1/logfile/tensorboard_bsz32,\
final_model:/home/gaokaifeng/project/OpenVoc-VidVRD/experiments/RelationCls_VidVRD/RePro/stage1/logfile/tensorboard_bsz16 \
  --port=6006 --bind_all

######## stage-2
tensorboard --logdir_spec=\
ablation_1:/home/gaokaifeng/project/OpenVoc-VidVRD/experiments/RelationCls_VidVRD/ablation_1/stage2/logfile/tensorboard_bsz32,\
ablation_2:/home/gaokaifeng/project/OpenVoc-VidVRD/experiments/RelationCls_VidVRD/ablation_2/stage2/logfile/tensorboard_bsz32,\
final_model:/home/gaokaifeng/project/OpenVoc-VidVRD/experiments/RelationCls_VidVRD/RePro/stage2/logfile/tensorboard_bsz16 \
  --port=6007 --bind_all

'''

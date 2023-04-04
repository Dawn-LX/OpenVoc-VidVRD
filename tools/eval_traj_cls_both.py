
import argparse
import os 
from tqdm import tqdm
from collections import defaultdict

import torch

from models.TrajClsModel_v2 import OpenVocTrajCls as OpenVocTrajCls_NoBgEmb
from models.TrajClsModel_v3 import OpenVocTrajCls as OpenVocTrajCls_0BgEmb


# from dataloaders.dataset_vidor_v3 import VidORTrajDataset
from dataloaders.dataset_vidvrd_v3 import VidVRDTrajDataset
from utils.utils_func import get_to_device_func,vIoU_broadcast
from utils.config_parser import parse_config_py
from utils.logger import LOGGER, add_log_to_file



def load_checkpoint(model,optimizer,scheduler,ckpt_path):
    checkpoint = torch.load(ckpt_path,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    scheduler.load_state_dict(checkpoint['sched_state_dict'])
    crt_epoch = checkpoint["crt_epoch"]
    batch_size = checkpoint["batch_size"]

    return model,optimizer,scheduler,crt_epoch,batch_size



def eval_TrajClsOpenVoc_bsz1(dataset_class,model_class,args,topks=[5,10]):
    cfg_path =  args.cfg_path
    ckpt_path = args.ckpt_path
    output_dir=args.output_dir
    eval_split = args.eval_split
    save_tag = args.save_tag

    if output_dir is None:
        output_dir = os.path.dirname(cfg_path)
    log_dir = os.path.join(output_dir,'logfile/')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_path = os.path.join(log_dir,f'eval_{save_tag}.log')
    add_log_to_file(log_path)

    configs = parse_config_py(cfg_path)
    dataset_cfg = configs["eval_dataset_cfg"]
    model_cfg = configs["model_cfg"]
    eval_cfg = configs["eval_cfg"]
    device = torch.device("cuda")

    if eval_split is None:
        assert dataset_cfg["class_splits"] is not None
    else:
        if eval_split == "base":
            class_splits = ("base",)
        elif eval_split == "novel":
            class_splits = ("novel",)
        elif eval_split == "all":
            class_splits = ("base","novel")
        else:
            assert False
        dataset_cfg["class_splits"] = class_splits

    LOGGER.info("dataset config: {}".format(dataset_cfg))
    LOGGER.info("model config: {}".format(model_cfg))
    LOGGER.info("evaluate config: {}".format(eval_cfg))

    vIoU_th     = eval_cfg["vIoU_th"]

    model = model_class(model_cfg,is_train=False)
    LOGGER.info(f"loading check point from {ckpt_path}")
    if not args.use_teacher:
        ckeck_point = torch.load(ckpt_path,map_location=torch.device('cpu'))
        state_dict = ckeck_point["model_state_dict"]
        model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    if hasattr(model,"reset_classifier_weights"):
        model.reset_classifier_weights(eval_split)


    LOGGER.info("preparing dataloader...")
    dataset = dataset_class(**dataset_cfg)

    collate_func = dataset.get_collator_func()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn = lambda x :x[0] ,
        num_workers = 0,
        drop_last= False,
        shuffle= False,
    )
    LOGGER.info("dataloader ready.")
    dataset_len = len(dataset)
    dataloader_len = len(dataloader)
    LOGGER.info(
        "len(dataset)=={},len(dataloader)=={}".format(dataset_len,dataloader_len)
    )

    LOGGER.info("start evaluating:")
    LOGGER.info("use config: {}".format(cfg_path))
    LOGGER.info("use device: {}; eval split: {}".format(device,eval_split))

    if isinstance(dataset,VidVRDTrajDataset):
        inference_for_vidvrd(model,device,dataloader,topks,vIoU_th)
    elif isinstance(dataset,VidORTrajDataset):
        inference_for_vidor(model,device,dataloader,topks,vIoU_th)

    LOGGER.info(f"log saved at {log_path}")
    LOGGER.handlers.clear()


def inference_for_vidor(model,device,dataloader,topks,vIoU_th):

    total_gt = 0
    total_hit = 0
    total_hit_at_k = defaultdict(int)
    for data in tqdm(dataloader):
        (
            video_name,
            traj_infos,
            traj_feats,
            traj_embds,   # include all segs in this video
            gt_annos,
            labels
        ) = data  # each data contains all segments in one video

        '''
        traj_feat & traj_embd has been filtered by traj_len_th & min_region_th
        traj_info = {
            "ids_left":torch.as_tensor(ids_left),  # (n_det,)
            "fstarts":torch.as_tensor(fstarts), # shape == (n_det,)
            "scores":torch.as_tensor(scores),  # shape == (n_det,)
            "bboxes":bboxes,  # list[tensor] , len== n_det, each shape == (num_boxes, 4)
        }

        gt_anno = {
            "labels":labels,    # shape == (num_traj,)
            "fstarts":fstarts,  # shape == (num_traj,)
            "bboxes":bboxes,    # len==num_traj, each shape == (num_bboxes,4)
        }
        '''
        n_trajs = [x.shape[0] for x in traj_embds]
        n_segs= len(n_trajs)
        traj_embds = torch.cat(traj_embds,dim=0)
        traj_feats = torch.cat(traj_feats,dim=0)

        with torch.no_grad():
            if args.use_teacher:
                traj_embds = traj_embds.to(device)
                cls_scores,cls_ids = model.forward_inference_bsz1(traj_embds,input_emb=True)
            else:
                traj_feats = traj_feats.to(device)
                cls_scores,cls_ids = model.forward_inference_bsz1(traj_feats,input_emb=False)
        cls_scores = cls_scores.cpu()
        cls_ids = cls_ids.cpu()
        
        cls_scores = torch.split(cls_scores,n_trajs,dim=0)
        cls_ids = torch.split(cls_ids,n_trajs,dim=0)

        for i in range(n_segs):
            gt_anno = gt_annos[i]
            traj_info = traj_infos[i]
            cs_ = cls_scores[i]
            ci_ = cls_ids[i]

            cs_, argids  = torch.sort(cs_,dim=0,descending=True)

            for k in topks:
                argids_topk = argids[:k]
                n_det,n_hit_at_k,n_gt = eval_traj_recall_topK_per_seg(traj_info,ci_,gt_anno,argids_topk,vIoU_th)
                total_hit_at_k[k] += n_hit_at_k
            total_gt += n_gt
    for k in topks:
        recall_at_k = total_hit_at_k[k] / total_gt
        LOGGER.info(f"total_hit_at_{k}={total_hit_at_k[k]},total_gt={total_gt},recall_at_{k}={recall_at_k}")
    


def inference_for_vidvrd(model,device,dataloader,topks,vIoU_th):

    total_gt = 0
    total_hit = 0
    total_hit_at_k = defaultdict(int)
    for data in tqdm(dataloader):
        (
            seg_tag,
            traj_infos,
            traj_feats,
            traj_embds,   
            gt_annos,
            labels
        ) = data  # 

        '''
        traj_feat & traj_embd has been filtered by traj_len_th & min_region_th
        traj_info = {
            "ids_left":torch.as_tensor(ids_left),  # (n_det,)
            "fstarts":torch.as_tensor(fstarts), # shape == (n_det,)
            "scores":torch.as_tensor(scores),  # shape == (n_det,)
            "bboxes":bboxes,  # list[tensor] , len== n_det, each shape == (num_boxes, 4)
        }

        gt_anno = {
            "labels":labels,    # shape == (num_traj,)
            "fstarts":fstarts,  # shape == (num_traj,)
            "bboxes":bboxes,    # len==num_traj, each shape == (num_bboxes,4)
        }
        '''
        with torch.no_grad():
            if args.use_teacher:
                traj_embds = traj_embds.to(device)
                cls_scores,cls_ids = model.forward_inference_bsz1(traj_embds,input_emb=True)
            else:
                traj_feats = traj_feats.to(device)
                cls_scores,cls_ids = model.forward_inference_bsz1(traj_feats,input_emb=False)
        cls_scores = cls_scores.cpu()
        cls_ids = cls_ids.cpu()
        cls_scores, argids  = torch.sort(cls_scores,dim=0,descending=True)
        for k in topks:
            argids_topk = argids[:k]
            n_det,n_hit_at_k,n_gt = eval_traj_recall_topK_per_seg(traj_infos,cls_ids,gt_annos,argids_topk,vIoU_th)
            total_hit_at_k[k] += n_hit_at_k
        total_gt += n_gt
    for k in topks:
        recall_at_k = total_hit_at_k[k] / total_gt
        LOGGER.info(f"total_hit_at_{k}={total_hit_at_k[k]},total_gt={total_gt},recall_at_{k}={recall_at_k}")
    


def eval_traj_recall_PosOnly(args):
    cfg_path =  args.cfg_path
    output_dir=args.output_dir
    eval_split = args.eval_split
    save_tag = args.save_tag

    if output_dir is None:
        output_dir = os.path.dirname(cfg_path)
    log_dir = os.path.join(output_dir,'logfile/')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_path = os.path.join(log_dir,f'eval_{save_tag}.log')
    add_log_to_file(log_path)

    configs = parse_config_py(cfg_path)
    dataset_cfg = configs["eval_dataset_cfg"]
    model_cfg = configs["model_cfg"]
    eval_cfg = configs["eval_cfg"]
    device = torch.device("cuda")

    if eval_split is None:
        assert dataset_cfg["class_splits"] is not None
    else:
        if eval_split == "base":
            class_splits = ("base",)
        elif eval_split == "novel":
            class_splits = ("novel",)
        elif eval_split == "all":
            class_splits = ("base","novel")
        else:
            assert False
        dataset_cfg["class_splits"] = class_splits

    LOGGER.info("dataset config: {}".format(dataset_cfg))
    LOGGER.info("model config: {}".format(model_cfg))
    LOGGER.info("evaluate config: {}".format(eval_cfg))

    vIoU_th     = eval_cfg["vIoU_th"]



    LOGGER.info("preparing dataloader...")
    dataset = VidORTrajDataset(**dataset_cfg)

    collate_func = dataset.get_collator_func()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn = lambda x :x[0] ,
        num_workers = 0,
        drop_last= False,
        shuffle= False,
    )
    LOGGER.info("dataloader ready.")
    dataset_len = len(dataset)
    dataloader_len = len(dataloader)
    LOGGER.info(
        "len(dataset)=={},len(dataloader)=={}".format(dataset_len,dataloader_len)
    )

    LOGGER.info("start evaluating:")
    LOGGER.info("use config: {}".format(cfg_path))
    LOGGER.info("eval split: {}".format(eval_split))


    total_gt = 0
    total_hit = 0
    for data in tqdm(dataloader):
        (
            video_name,
            traj_infos,
            traj_feats,
            traj_embds,
            gt_annos,
            labels
        ) = data  # each data contains all segments in one video

        '''
        traj_feat & traj_embd has been filtered by traj_len_th & min_region_th
        traj_info = {
            "ids_left":torch.as_tensor(ids_left),  # (n_det,)
            "fstarts":torch.as_tensor(fstarts), # shape == (n_det,)
            "scores":torch.as_tensor(scores),  # shape == (n_det,)
            "bboxes":bboxes,  # list[tensor] , len== n_det, each shape == (num_boxes, 4)
        }

        gt_anno = {
            "labels":labels,    # shape == (num_traj,)
            "fstarts":fstarts,  # shape == (num_traj,)
            "bboxes":bboxes,    # len==num_traj, each shape == (num_bboxes,4)
        }
        '''
        n_trajs = [x.shape[0] for x in traj_embds]
        n_segs= len(n_trajs)
        
        cls_ids = torch.ones(size=(sum(n_trajs),))
        
        cls_ids = torch.split(cls_ids,n_trajs,dim=0)

        for i in range(n_segs):
            gt_anno = gt_annos[i]
            traj_info = traj_infos[i]
            ci_ = cls_ids[i]

            n_det,n_hit,n_gt = eval_traj_recall_per_seg(traj_info,ci_,gt_anno,vIoU_th,use_cls=False)
            total_hit += n_hit
            total_gt += n_gt
    recall = total_hit / total_gt
    LOGGER.info(f"total_hit={total_hit},total_gt={total_gt},recall={recall}")
    
    LOGGER.info(f"log saved at {log_path}")
    LOGGER.handlers.clear()

def eval_traj_recall_per_seg(det_info,det_cls_ids,gt_anno,vIoU_th,use_cls=True):
    if gt_anno is None:
        return 0,0,0    
    det_trajs = det_info["bboxes"]    # list[tensor] , len== n_det, each shape == (num_boxes, 4)
    n_det = len(det_trajs)
    det_fstarts = det_info["fstarts"]  # (n_det,)
    
    gt_trajs = gt_anno["bboxes"]      # list[tensor] , len== n_gt,  each shape == (num_boxes, 4)
    gt_fstarts = gt_anno["fstarts"]   # (n_gt,)
    gt_labels = gt_anno["labels"]     # (n_gt,)
    n_gt = len(gt_labels)

    

    viou_matrix = vIoU_broadcast(det_trajs,gt_trajs,det_fstarts,gt_fstarts)  # (n_det, n_gt)
    '''
    e.g.,
    viou_matrix=[[0.0000, 0.0000, 0.0000],
                [0.0146, 0.0121, 0.0124],
                [0.0879, 0.1795, 0.2318],
                [0.3935, 0.0000, 0.2242]], with vIoU_th = 0.15
    '''
    if use_cls:
        cls_eq_mask = det_cls_ids[:,None] == gt_labels[None,:]  # (n_det,n_gt)
        viou_matrix[~cls_eq_mask] = 0.0

    max_vious, gt_ids = torch.max(viou_matrix,dim=-1)  # shape == (n_det,)
    mask = max_vious > vIoU_th
    gt_ids[~mask] = -1
    hit_gt_ids = list(set(gt_ids.tolist()))  # range: -1, 0 ~ n_gt-1
    
    n_hit = (torch.as_tensor(hit_gt_ids) >=0).sum().item()
    # print("n_det,n_hit, n_gt",n_det,n_hit, n_gt)
    return n_det,n_hit, n_gt

def eval_traj_recall_topK_per_seg(det_info,det_cls_ids,gt_anno,ids_topk,vIoU_th,traj_ids=None):
    
    if gt_anno is None:
        return 0,0,0

    det_trajs = det_info["bboxes"]    # list[tensor] , len== n_det, each shape == (num_boxes, 4)
    n_det = len(det_trajs)
    det_fstarts = det_info["fstarts"]  # (n_det,)
    if traj_ids is not None:
        # `det_cls_ids` has been filtered outside if `traj_ids is not None`
        det_fstarts = det_fstarts[traj_ids]
        det_trajs = [det_trajs[idx] for idx in traj_ids.tolist()]

    gt_trajs = gt_anno["bboxes"]      # list[tensor] , len== n_gt,  each shape == (num_boxes, 4)
    gt_fstarts = gt_anno["fstarts"]   # (n_gt,)
    gt_labels = gt_anno["labels"]     # (n_gt,)
    n_gt = len(gt_labels)

    #### select topk
    det_trajs = [det_trajs[idx] for idx in ids_topk.tolist()]
    # print(type(det_fstarts))
    det_fstarts = det_fstarts[ids_topk]
    det_cls_ids = det_cls_ids[ids_topk]
    ####

    viou_matrix = vIoU_broadcast(det_trajs,gt_trajs,det_fstarts,gt_fstarts)  # (n_det, n_gt)
    '''
    e.g.,
    viou_matrix=[[0.0000, 0.0000, 0.0000],
                [0.0146, 0.0121, 0.0124],
                [0.0879, 0.1795, 0.2318],
                [0.3935, 0.0000, 0.2242]], with vIoU_th = 0.15
    '''
    cls_eq_mask = det_cls_ids[:,None] == gt_labels[None,:]  # (n_det,n_gt)
    viou_matrix[~cls_eq_mask] = 0.0

    max_vious, gt_ids = torch.max(viou_matrix,dim=-1)  # shape == (n_det,)
    mask = max_vious > vIoU_th
    gt_ids[~mask] = -1
    hit_gt_ids = list(set(gt_ids.tolist()))  # range: -1, 0 ~ n_gt-1
    n_hit = (torch.as_tensor(hit_gt_ids) >=0).sum().item()

    return n_det,n_hit, n_gt


if __name__ == "__main__":
    # dataloader_demo()
    
    # assert False
    
    parser = argparse.ArgumentParser(description="Object Detection Demo")
    
    parser.add_argument("--model_class", type=str,help="...")
    parser.add_argument("--dataset_class", type=str,help="...")
    
    parser.add_argument("--cfg_path", type=str,help="...")
    parser.add_argument("--ckpt_path", type=str,help="...")
    parser.add_argument("--output_dir", type=str, help="...") # 如果不加 --output_dir， 缺省值就是None
    parser.add_argument("--save_tag", type=str,default="",help="...")
    parser.add_argument("--eval_split", type=str,default="novel",help="...")
    parser.add_argument("--use_teacher", action="store_true")
     
    args = parser.parse_args()

    # for split in ["base","novel","all"]:
    model_class = eval(args.model_class)
    dataset_class = eval(args.dataset_class)
    # eval_traj_recall_PosOnly(args)
    eval_TrajClsOpenVoc_bsz1(
        dataset_class=dataset_class,
        model_class = model_class,
        args = args
    )
    # OpenVocTrajCls_0BgEmb
    # OpenVocTrajCls_NoBgEmb
    # VidORTrajDataset
    # VidVRDTrajDataset
    
    '''
    ### Table-1 Alpro OpenVocTrajCls_0BgEmb or OpenVocTrajCls_NoBgEmb either one is fine
    CUDA_VISIBLE_DEVICES=1 python tools/eval_traj_cls_both.py \
        --dataset_class VidVRDTrajDataset \
        --model_class OpenVocTrajCls_0BgEmb \
        --cfg_path  experiments/TrajCls_VidVRD/0BgEmb/cfg_.py \
        --eval_split novel \
        --output_dir experiments/TrajCls_VidVRD/ALPro \
        --use_teacher \
        --save_tag teacher_novel
    
    ### Table-1 RePro-#1 w/o Distil & w/o BgEmb OpenVocTrajCls_NoBgEmb
    CUDA_VISIBLE_DEVICES=1 python tools/eval_traj_cls_both.py \
        --dataset_class VidVRDTrajDataset \
        --model_class OpenVocTrajCls_NoBgEmb \
        --cfg_path   experiments/TrajCls_VidVRD/NoBgEmb/cfg_.py \
        --ckpt_path  experiments/TrajCls_VidVRD/NoBgEmb/model_final_wo_distil_bs128_epoch_50.pth \
        --eval_split novel \
        --output_dir experiments/TrajCls_VidVRD/NoBgEmb  \
        --save_tag wo_distil_novel
    

    ### Table-1 RePro-#2 w/o Distil & w/ BgEmb OpenVocTrajCls_0BgEmb
    CUDA_VISIBLE_DEVICES=1 python tools/eval_traj_cls_both.py \
        --dataset_class VidVRDTrajDataset \
        --model_class OpenVocTrajCls_0BgEmb \
        --cfg_path   experiments/TrajCls_VidVRD/0BgEmb/cfg_.py \
        --ckpt_path  experiments/TrajCls_VidVRD/0BgEmb/model_final_wo_distil_bs128_epoch_50.pth  \
        --eval_split novel \
        --output_dir experiments/TrajCls_VidVRD/0BgEmb  \
        --save_tag wo_distil_novel
    

    ### Table-1 RePro-#3 w/ Distil & w/ BgEmb OpenVocTrajCls_0BgEmb
    CUDA_VISIBLE_DEVICES=1 python tools/eval_traj_cls_both.py \
        --dataset_class VidVRDTrajDataset \
        --model_class OpenVocTrajCls_0BgEmb \
        --cfg_path   experiments/TrajCls_VidVRD/0BgEmb/cfg_.py \
        --ckpt_path  experiments/TrajCls_VidVRD/0BgEmb/model_final_with_distil_w5bs128_epoch_50.pth  \
        --eval_split novel \
        --output_dir experiments/TrajCls_VidVRD/0BgEmb  \
        --save_tag with_distil_novel

    ### Table-1 RePro-#4 w/ Distil & w/o BgEmb OpenVocTrajCls_NoBgEmb
    CUDA_VISIBLE_DEVICES=1 python tools/eval_traj_cls_both.py \
        --dataset_class VidVRDTrajDataset \
        --model_class OpenVocTrajCls_NoBgEmb \
        --cfg_path    experiments/TrajCls_VidVRD/NoBgEmb/cfg_.py \
        --ckpt_path    experiments/TrajCls_VidVRD/NoBgEmb/model_final_with_distil_w5bs128_epoch_50.pth \
        --eval_split novel \
        --output_dir   experiments/TrajCls_VidVRD/NoBgEmb \
        --save_tag with_distil_novel
    

    '''
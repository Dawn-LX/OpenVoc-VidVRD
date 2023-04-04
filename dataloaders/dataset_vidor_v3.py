
import os
import json
import math
import pickle
from collections import defaultdict
from easydict import EasyDict
from copy import deepcopy,copy

import numpy as np
import torch
from tqdm import tqdm

from utils.utils_func import vIoU_broadcast,vPoI_broadcast,trajid2pairid,temporal_overlap,bbox_GIoU
from utils.logger import LOGGER
def load_json(filename):
    with open(filename, "r") as f:
        x = json.load(f)
    return x



def _reset_dataset_split(split):
    train = {x:"training" for x in ["train","training"]}
    val = {x:"validation" for x in ["val","validation"]}
    split_dict = {}
    for x in [train,val]:
        split_dict.update(x)
    
    return split_dict[split.lower()]

def _to_xywh(bboxes):
    x = (bboxes[...,0] + bboxes[...,2])/2
    y = (bboxes[...,1] + bboxes[...,3])/2
    w = bboxes[...,2] - bboxes[...,0]
    h = bboxes[...,3] - bboxes[...,1]
    return x,y,w,h



def get_relative_position_feature(traj_fstarts,traj_bboxes,sids,oids):
    
    # traj_fstarts  # (n_det,)
    # traj_bboxes # list[tensor] , len == n_det, each shape == (num_boxes, 4)     
    
    ## 1.
    s_trajs = [traj_bboxes[idx] for idx in sids]  # format: xyxy
    o_trajs = [traj_bboxes[idx] for idx in oids]  # len == n_pair, each shape == (n_frames,4)

    s_fstarts = traj_fstarts[sids]  # (n_pair,)
    o_fstarts = traj_fstarts[oids]  # 

    s_lens = torch.as_tensor([x.shape[0] for x in s_trajs],device=s_fstarts.device)  # (n_pair,)
    o_lens = torch.as_tensor([x.shape[0] for x in o_trajs],device=o_fstarts.device)

    s_duras = torch.stack([s_fstarts,s_fstarts+s_lens],dim=-1)  # (n_pair,2)
    o_duras = torch.stack([o_fstarts,o_fstarts+o_lens],dim=-1)  # (n_pair,2)

    s_bboxes = [torch.stack((boxes[0,:],boxes[-1,:]),dim=0) for boxes in s_trajs]  # len == n_pair, each shape == (2,4)
    s_bboxes = torch.stack(s_bboxes,dim=0)  # (n_pair, 2, 4)  # 2 stands for the start & end bbox of the traj

    o_bboxes = [torch.stack((boxes[0,:],boxes[-1,:]),dim=0) for boxes in o_trajs]
    o_bboxes = torch.stack(o_bboxes,dim=0)  # (n_pair, 2, 4)


    ## 2. calculate relative position feature
    subj_x, subj_y, subj_w, subj_h = _to_xywh(s_bboxes.float())  # (n_pair,2)
    obj_x, obj_y, obj_w, obj_h = _to_xywh(o_bboxes.float())      # (n_pair,2)

    log_subj_w, log_subj_h = torch.log(subj_w), torch.log(subj_h)
    log_obj_w, log_obj_h = torch.log(obj_w), torch.log(obj_h)

    rx = (subj_x-obj_x)/obj_w   # (n_pair,2)
    ry = (subj_y-obj_y)/obj_h
    rw = log_subj_w-log_obj_w
    rh = log_subj_h-log_obj_h
    ra = log_subj_w+log_subj_h-log_obj_w-log_obj_h
    rt = (s_duras-o_duras) / 30  # (n_pair,2)
    rel_pos_feat = torch.cat([rx,ry,rw,rh,ra,rt],dim=-1)  # (n_pair,12)

    return rel_pos_feat
        
def _get_traj_pair_GIoU(traj_bboxes,sids,oids):

    n_pair = len(sids)
    s_trajs = [traj_bboxes[idx] for idx in sids]  # format: xyxy
    o_trajs = [traj_bboxes[idx] for idx in oids]  # len == n_pair, each shape == (n_frames,4)


    start_s_box = torch.stack([boxes[0,:] for boxes in s_trajs],dim=0)  # (n_pair, 4)
    start_o_box = torch.stack([boxes[0,:] for boxes in o_trajs],dim=0)  # (n_pair, 4)

    end_s_box = torch.stack([boxes[-1,:] for boxes in s_trajs],dim=0)  # (n_pair, 4)
    end_o_box = torch.stack([boxes[-1,:] for boxes in o_trajs],dim=0)  # (n_pair, 4)

    start_giou = bbox_GIoU(start_s_box,start_o_box)[range(n_pair),range(n_pair)]  # (n_pair,)
    end_giou = bbox_GIoU(end_s_box,end_o_box)[range(n_pair),range(n_pair)]  # (n_pair,)
    se_giou = torch.stack([start_giou,end_giou],dim=-1)  # (n_pair,2)

    return se_giou

def _remove_HeadBasePred_video_names(cls2split):
    '''
    Here we remove those videos which only has head base predicate classes
    refer to func:`stat_videoname2pred` in `visualization_vidor/stat_triplet_bias.py`
    '''
    HEAD_TOPK = 4
    filename = "/home/gkf/project/VidVRD_VidOR/statistics/VidORtrain_triplet_bias.json" # refer to /home/gkf/project/VidVRD_VidOR/statistics/stat_triplet_bias.py
    video_triplets = load_json(filename)

    videoname2pred = defaultdict(set)
    pred2counts_base = defaultdict(int)
    for video_name, triplet_list in video_triplets.items():
        for spo in triplet_list:
            p = spo[1]
            if cls2split[p] == "base":
                videoname2pred[video_name].add(p)
                pred2counts_base[p] += 1


    sorted_pred2counts_base = sorted(pred2counts_base.items(),key=lambda x:x[1],reverse=True)


    sorted_pred_base = [x[0] for x in sorted_pred2counts_base]
    head_preds = sorted_pred_base[:HEAD_TOPK]
    video_name_list = set(videoname2pred.keys())
    print(len(video_name_list))

    for video_name,pred_set in videoname2pred.items():
        is_all_head = all(p in head_preds for p in list(pred_set))
        if is_all_head:
            video_name_list.remove(video_name)
    

    print(len(video_name_list))

    # pred2counts_base = defaultdict(int)
    # for video_name in video_name_list:
    #     triplet_list = video_triplets[video_name]
    #     for spo in triplet_list:
    #         p = spo[1]
    #         if cls2split[p] == "base":
    #             pred2counts_base[p] += 1
    
    # sorted_pred2counts_base = sorted(pred2counts_base.items(),key=lambda x:x[1],reverse=True)
    # print(sorted_pred2counts_base)

    return video_name_list

class VidORTrajDataset(object):
    '''
    this version do not use dict to pre-save data in memory
    '''
    def __init__(self,
        class_splits,
        dataset_split,
        class_spilt_info_path = "configs/VidOR_OjbectClass_spilt_info_v2.json",
        dataset_dir = "/home/gkf/project/VidVRD_VidOR/vidor-dataset",
        tracking_res_dir = {
            "train":"/home/gkf/project/VidVRD-II/tracklets_results/VidORtrainVideoLevel_tracking_results_th-15-5",
            "val":"/home/gkf/project/VidVRD-II/tracklets_results/VidORvalVideoLevel_tracking_results_th-15-5"
        },
        traj_features_dir={
            "train":"/home/gkf/project/scene_graph_benchmark/output/VidORtrain_traj_features_th-15-5",
            "val":"/home/gkf/project/scene_graph_benchmark/output/VidORval_traj_features",
        },
        traj_embds_dir = {
            "train":"/home/gkf/project/ALPRO/extract_features_output/VidOR_TrajFeatures256_th-15-5",
            "val":"/home/gkf/project/ALPRO/extract_features_output/VidORval_TrajFeatures256",
        },
        cache_dir = "datasets/cache",
        num_sample_segs = 32,
        vIoU_th = 0.5,
        assign_labels = None, # None means not specified
        subset_idx_range = [0,7000],
    ):
        super().__init__()
        self.vIoU_th = vIoU_th
        self.num_sample_segs = num_sample_segs
        self.class_splits = tuple(cs.lower() for cs in class_splits)   # e.g., ("base","novel"), or ("base",) or ("novel",)
        self.dataset_split = dataset_split.lower()  
        assert self.dataset_split in ("train","val")
        if self.dataset_split == "val":
            subset_idx_range = [0,835]
        self.subset_idx_range = subset_idx_range
        if assign_labels is None:
            assign_labels = self.dataset_split == "train"
        

        with open(class_spilt_info_path,'r') as f:
            self.class_split_info = json.load(f)
        self.traj_cls2id_map = self.class_split_info["cls2id"]
        self.traj_cls2spilt_map = self.class_split_info["cls2split"]

        self.dataset_dir = dataset_dir
        self.cache_dir  = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.tracking_res_dir = tracking_res_dir[self.dataset_split]
        # .../tracklets_results/VidORtrain_tracking_results_th-15-5/1010_8872539414/1010_8872539414-0825-0855.json
        self.traj_features_dir = traj_features_dir[self.dataset_split]
        self.traj_embds_dir = traj_embds_dir[self.dataset_split]
        
        self.prepare_segment_tags(subset_idx_range)


        if self.dataset_split == "train" and len(self.video_names) != 7000:
            sid,eid = subset_idx_range
            txt_ = "subset range [{}:{}] ;".format(sid,eid)
        else:
            txt_ = ""
        LOGGER.info("{} num videos:{}".format(txt_,len(self.video_names)))


        if self.dataset_split == "train" and assign_labels:
            ### NOTE only base labels
            assert self.class_splits == ("base",)
            # assert subset_idx_range[0] == 0 and subset_idx_range[1] == 7000  # use other cached label if specify subset
            if list(subset_idx_range) == [0,7000]:
                path_ = os.path.join(self.cache_dir,"VidORtrain_DetTrajAssignedLabels_vIoU-{:.2f}_all7000.pth".format(self.vIoU_th))
            else:
                sid,eid = subset_idx_range
                path_ = os.path.join(self.cache_dir, "VidORtrain_DetTrajAssignedLabels_vIoU-{:.2f}_{}-{}.pth".format(vIoU_th,sid,eid))

            if os.path.exists(path_):
                LOGGER.info("label_infos loading from {}".format(path_))
                label_infos = torch.load(path_)
            else:
                LOGGER.info("no cache file found, assigning labels...")
                label_infos = self.label_assignment()
                torch.save(label_infos,path_)
                LOGGER.info("label_infos saved at {}".format(path_))
            LOGGER.info("label info load Done.")

            all_labels = label_infos["all_labels"]  # (N_det,)
            labeled_seg_ids = label_infos["labeled_seg_ids"]  # (n_labeled_segs,) seg_idx w.r.t all-7000
            n_trajs_per_seg = label_infos["n_trajs_per_seg"]  # (n_labeled_segs,)
            assert torch.all(all_labels <= 50)
            all_labels = torch.split(all_labels,n_trajs_per_seg.tolist(),dim=0)

            self.labeled_seg_absids = labeled_seg_ids # NOTE  seg_idx w.r.t all-7000
            # here we has filtered out segs with no labels and segs with n_det_traj = 0

            self.assigned_labels = all_labels  # len == N_seg, each shape == (n_det,)
            # del self.traj_annos  # for train we only use self.labels, del self.traj_annos (load `traj_annos` in `self.label_assignment()`)
        
        
        LOGGER.info("---------- dataset constructed, len(self) == {}".format(len(self)))

    def __len__(self):

        return len(self.video_names)


    def prepare_segment_tags(self,subset_idx_range):
        self.anno_dir = os.path.join(self.dataset_dir,"annotation",_reset_dataset_split(self.dataset_split)) # .../vidor-dataset/training
        group_ids = os.listdir(self.anno_dir)
        video_names_all = []
        for gid in group_ids:
            filenames = os.listdir(os.path.join(self.anno_dir,gid))
            video_names_all  += [gid + "_" + filename.split(".")[0]  for filename in filenames]

        video_names_all = sorted(video_names_all)
        sid,eid = subset_idx_range
        video_names = video_names_all[sid:eid]
        
        if self.dataset_split == "train":
            # fix this, TODO : load video-level .json and get segment_tags_all
            tmp = "/home/gkf/project/VidVRD-II/tracklets_results/VidORtrain_tracking_results_th-15-5"  
        else:
            tmp = "/home/gkf/project/VidVRD-II/tracklets_results/VidORval_tracking_results_th-15-5"
        video2seg = defaultdict(list)
        segment_tags_all = []
        for video_name in video_names_all:
            seg_filenames = sorted(os.listdir(os.path.join(tmp,video_name)))
            for filename in seg_filenames:
                seg_tag = filename.split('.')[0] # e.g., 1010_8872539414-0825-0855
                segment_tags_all.append(seg_tag)
                video2seg[video_name].append(seg_tag)
                

        seg2relidx = dict()
        for video_name, seg_tags in video2seg.items():
            for rel_idx,seg_tag in enumerate(seg_tags):
                seg2relidx[seg_tag] = rel_idx

        # TODO add self.segment_tags according to video_names (subset) for label assignment 
        # （label assign完成之后 这个代码改过了）， 要复现的话，需要重新增加 self.segment_tags (用在VidORTrajDataset_ForAssignLabels 中)

        self.seg2absidx = {seg_tag:idx for idx,seg_tag in enumerate(segment_tags_all)}
        self.seg2relidx = seg2relidx
        self.video2seg = video2seg
        self.video_names = video_names
        self.video_names_all = video_names_all
        self.segment_tags_all = segment_tags_all

    def get_traj_infos(self,video_name,segment_tags,rt_ntrajs_only=True):
        # we have filter originally saved tracking results,
        # refer to func:`filter_track_res_and_feature` in `/home/gkf/project/VidVRD-II/video_object_detection/object_bbox2traj_segment.py`
        '''
        tracking_results = {seg_tag1:res_1,seg_tag2:res_2,...,"n_trajs":[16,15,24,0,14,...]}
        res_i = [
            {
                'fstarts': int(fstart),     # relative frame idx w.r.t this segment
                'score': score,             # -1 for gt
                'bboxes': bboxes.tolist()   # list[list], len == num_frames, format xyxy

                ### for det traj
                'label':  cls_id            # int, w.r.t VinVL classes
                
                #### for gt_traj in train-set
                'class':  class_name        # str    
                'tid':tid                   # int, original tid in annotation (w.r.t video range)
                
            },
            ...
        ]  # len(res_i) == num_tracklets
        '''
        
        path = os.path.join(self.tracking_res_dir,video_name+".json")
        with open(path,'r') as f:
            tracking_results = json.load(f)

        rel_ids = [self.seg2relidx[seg_tag] for seg_tag in segment_tags]
        n_trajs_b4sp = tracking_results["n_trajs"]  # `b4sp` means before sample
        n_trajs = [n_trajs_b4sp[idx] for idx in rel_ids]
        if rt_ntrajs_only:
            return None,n_trajs,n_trajs_b4sp

        res0 = tracking_results[segment_tags[0]][0]
        # print(res0.keys())
        has_cls =  "class" in res0.keys()
        has_tid = "tid" in res0.keys()
        # print(has_cls)

        traj_infos = []
        for seg_tag in segment_tags:
            fstarts = []
            scores = []
            bboxes = []
            cls_ids = []
            tids = []
            for res in tracking_results[seg_tag]:  # this loop can be empty
                assert isinstance(res["fstart"],int), "seg_tag={}".format(seg_tag)
                fstarts.append(res["fstart"])  # relative frame_id  w.r.t segment fstart
                scores.append(res["score"])  # score obtained from object detetcor(FasterRCNN), we not use this score in relation-classification
                bboxes.append(torch.as_tensor(res["bboxes"])) # (num_boxes, 4)
                if has_cls:
                    cls_ids.append(self.traj_cls2id_map[res["class"]]) 
                if has_tid:
                    tids.append(res["tid"])
            
            traj_info = {
                "fstarts":torch.as_tensor(fstarts), # shape == (n_det,)
                "scores":torch.as_tensor(scores),  # shape == (n_det,)
                "bboxes":bboxes,  # list[tensor] , len== n_det, each shape == (num_boxes, 4) 1025_11664231455
                # "VinVL_clsids":torch.as_tensor(cls_ids),  # shape == (n_det,)
            }
            if has_cls:
                traj_info.update({"cls_ids":torch.as_tensor(cls_ids)})  # len == n_det
                # print(cls_ids)
            if has_tid:
                traj_info.update({"tids":torch.as_tensor(tids)})
            
            traj_infos.append(traj_info)



        return traj_infos,n_trajs,n_trajs_b4sp

    def get_traj_features(self,video_name,segment_tags,n_trajs):
        
        # relative_idx = self.seg2relidx[seg_tag]
        rel_ids = [self.seg2relidx[seg_tag] for seg_tag in segment_tags]
        rel_ids = torch.as_tensor(rel_ids)  # (n_seg,)

        #### traj RoI features (2048-d)
        # /home/gkf/project/scene_graph_benchmark/output/VidORtrain_gt_traj_features/0000_2401075277_traj_features.npy
        path = os.path.join(self.traj_features_dir,video_name+"_traj_features.npy")
        traj_features = np.load(path) 
        traj_features = torch.from_numpy(traj_features).float()  # float32, # (N_traj,2048)  
        ### TODO add relative seg_ids, i.e, shape == (N_traj,2049)
        # seg_rel_ids = traj_features[:,0]
        # mask = seg_rel_ids.type(torch.long) == rel_ids  # (N_traj,n_seg)
        # mask = torch.any(mask,dim=-1)  # (N_traj)
        # traj_features = traj_features[mask,:]

        traj_features = torch.split(traj_features,n_trajs,dim=0)  # len == N_seg (before sample)
        traj_features = [traj_features[idx] for idx in rel_ids]   # len == n_seg (after sample)
        
        #### traj Alpro-embeddings (256-d)
        # /home/gkf/project/ALPRO/extract_features_output/VidOR_TrajFeatures256_gt/0000_2401075277.pth
        # (N_traj,258) format of each line: [seg_id,tid,256d-feature] (tid w.r.t segment range (before traj_len filter), not original anno)
        
        path = os.path.join(self.traj_embds_dir,video_name+".pth")
        tmp = torch.load(path)  # (N_traj,258)
        seg_rel_ids = tmp[:,0]  # (N_traj,)
        tids_aft_filter = tmp[:,1]  #this is deprecated, because we apply traj_len_th filter directly after the Seq-NMS and save the tracking results .json file
        traj_embds = tmp[:,2:]
        # mask = seg_rel_ids.type(torch.long) == rel_ids  # (N_traj,n_seg)
        # mask = torch.any(mask,dim=-1)  # (N_traj)
        # traj_embds = traj_embds[mask,:]
        traj_embds = torch.split(traj_embds,n_trajs,dim=0)
        traj_embds = [traj_embds[idx] for idx in rel_ids]


        return traj_features,traj_embds


    def get_annos(self,video_name,segment_tags):
    
        # LOGGER.info("preparing annotations for data_split: {}, class_splits: {} ".format(self.dataset_split,self.class_splits))


        gid,vid = video_name.split('_') 
        anno_path = os.path.join(self.anno_dir,gid,vid+".json")
        # .../vidor-dataset/annotation/training/0001/3058613626.json
        with open(anno_path,'r') as f:
            video_annos = json.load(f)  # refer to `datasets/vidvrd-dataset/format.py`


        traj_annos_this_video = []
        for seg_tag in segment_tags:
            video_name, fstart, fend = seg_tag.split('-')  # e.g., "ILSVRC2015_train_00010001-0015-0045"
            fstart,fend = int(fstart),int(fend)
            trajid2cls_map = {traj["tid"]:traj["category"] for traj in video_annos["subject/objects"]}
            # e.g., {0: 'bicycle', 1: 'bicycle', 2: 'person', 3: 'person', 5: 'person'} # not necessarily continuous

            trajs_info = {tid:defaultdict(list) for tid in trajid2cls_map.keys()}      
            annotated_len = len(video_annos["trajectories"])
            
            for frame_id in range(fstart,fend,1):  # 75， 105
                if frame_id >= annotated_len:  
                    # e.g., for "ILSVRC2015_train_00008005-0075-0105", annotated_len==90, and anno_per_video["frame_count"] == 130
                    break

                frame_anno = video_annos["trajectories"][frame_id]  # NOTE frame_anno can be [] (empty) for all `fstart` to `fend`
                # LOGGER.info(seg_tag,frame_id,len(frame_anno))
                for bbox_anno in frame_anno:  
                    tid = bbox_anno["tid"]
                    bbox = bbox_anno["bbox"]
                    bbox = [bbox["xmin"],bbox["ymin"],bbox["xmax"],bbox["ymax"]]
                    trajs_info[tid]["bboxes"].append(bbox)
                    trajs_info[tid]["frame_ids"].append(frame_id)

            labels = []
            fstarts = []
            bboxes = []
            for tid, info in trajs_info.items():
                if not info:  # i.e., if `info` is empty, we continue
                    continue
                class_ = trajid2cls_map[tid]
                split_ = self.traj_cls2spilt_map[class_]
                if not (split_ in self.class_splits):
                    continue
                
                labels.append(
                    self.class_split_info["cls2id"][class_]
                )
                fstarts.append(
                    min(info["frame_ids"]) - fstart  # relative frame_id  w.r.t segment fstart
                )
                bboxes.append(
                    torch.as_tensor(info["bboxes"])  # shape == (num_bbox,4)
                )

            if labels:
                labels = torch.as_tensor(labels)  
                fstarts = torch.as_tensor(fstarts)  
                traj_annos = {
                    "labels":labels,    # shape == (num_traj,)
                    "fstarts":fstarts,  # shape == (num_traj,)
                    "bboxes":bboxes,    # len==num_traj, each shape == (num_bboxes,4)
                }
            else:
                traj_annos = None
            
            traj_annos_this_video.append(traj_annos)
        return traj_annos_this_video
                

    
    def __getitem__(self,idx):
        video_name = deepcopy(self.video_names[idx])  # return video_name for debug
        segment_tags  = self.video2seg[video_name]

        if self.dataset_split == "train":
            num_sample_segs = min(self.num_sample_segs,len(segment_tags))
            segment_tags = np.random.choice(segment_tags,num_sample_segs,replace=False)
        
            labels,labeled_sample_ids = self.get_labels(segment_tags)  #
            if len(labels) == 0:
                rand_idx = np.random.choice(list(range(len(self))))
                # LOGGER.info("video: {} with num_sample_segs={}, has no labeled segments".format(video_name,num_sample_segs))
                return self.__getitem__(rand_idx)
                

        traj_infos,n_trajs,n_trajs_b4sp = self.get_traj_infos(video_name,segment_tags,rt_ntrajs_only=self.dataset_split=="train") # b4sp means before sample
        traj_features,traj_embds = self.get_traj_features(video_name,segment_tags,n_trajs_b4sp)
        assert len(n_trajs) == len(traj_features) and len(n_trajs) == len(traj_embds)
        for nt,tf,te in zip(n_trajs,traj_features,traj_embds):
            assert tf.shape[0] == te.shape[0] and nt == te.shape[0]
        

        if self.dataset_split == "val":
            gt_annos = self.get_annos(video_name,segment_tags)
            labels = None
        else:
            assert traj_infos is None
            gt_annos = None
            
            # print(labeled_sample_ids,len(traj_features))
            traj_features = [traj_features[idx] for idx in labeled_sample_ids]
            traj_embds = [traj_embds[idx] for idx in labeled_sample_ids]
            labels = torch.cat(labels,dim=0)        
            traj_features = torch.cat(traj_features,dim=0)  # (N_traj,) total trajs in these sampled segs
            traj_embds = torch.cat(traj_embds,dim=0)
            assert traj_embds.shape[0] == labels.shape[0]
        
        
        return video_name,traj_infos,traj_features,traj_embds,gt_annos,labels

    def get_labels(self,segment_tags):
        # self.labeled_seg_ids  shape == (N_labeled_seg,)
        # self.assigned_labels shape == (N_labeled_seg,)
        # NOTE we must use relative_id w.r.t labeled segs (not w.r.t all segments) to index self.assigned_labels
        # NOTE some sampled segment might not be labeled

        sampled_ids = [self.seg2absidx[seg_tag] for seg_tag in segment_tags]
        sampled_ids = torch.as_tensor(sampled_ids)  # (n_sampled,)
        mask  = self.labeled_seg_absids[:,None] == sampled_ids[None,:]   # (N_labeled_seg,n_sampled)
        mask0 = torch.any(mask,dim=-1)  # (N_labeled_seg,)
        labeled_seg_relids = mask0.nonzero(as_tuple=True)[0].tolist()

        mask1 = torch.any(mask,dim=0)  # (n_sampled,)
        # print(mask1.shape)
        labeled_sample_ids = mask1.nonzero(as_tuple=True)[0].tolist()

        labels = [self.assigned_labels[idx]   for idx in labeled_seg_relids]
        # len(labels) < n_sampled when some sampled segs are not labeled
        # labels[i].sahpe == (n_traj,)

        return labels,labeled_sample_ids


    def get_collator_func(self):
        
        def collator_func(batch_data):
            '''
            this collator_func simply swaps the order of inner and outer of batch_data
            seg_tag, traj_info,traj_feat,traj_embd, gt_anno, labels=batch_data[0]
            '''

            bsz = len(batch_data)
            num_rt_vals = len(batch_data[0])
            return_values = tuple([batch_data[bid][vid] for bid in range(bsz)] for vid in range(num_rt_vals))
            return return_values

        return collator_func


    def label_assignment(self):
        '''
        TODO FIXME write this func in a multi-process manner
        currently we use `class VidVRDUnifiedDatasetForLabelAssign` 
        and wrap it using torch's DataLoader to assign label in a multi-process manner 
        '''
        LOGGER.info("please use `class VidVRDUnifiedDatasetForLabelAssign` to pre-assign label and save as cache")
        raise NotImplementedError


class VidORTrajDataset_ForAssignLabels(VidORTrajDataset):
    def __init__(self,**kargs):
        kargs["assign_labels"] = False
        super().__init__(**kargs)
    
    def __getitem__(self, idx):

        seg_tag = self.segment_tags[idx]   # return seg_tag for debug
        
        det_info = self.traj_infos[seg_tag]
        det_trajs = det_info["bboxes"]    # list[tensor] , len== n_det, each shape == (num_boxes, 4)
        # this can be empty list
        if len(det_trajs) == 0:
            return deepcopy(seg_tag),None
        det_fstarts = det_info["fstarts"]  # (n_det,)

        gt_anno = self.traj_annos[seg_tag]
        if gt_anno is None:
            return deepcopy(seg_tag),None

        gt_trajs = gt_anno["bboxes"]      # list[tensor] , len== n_gt,  each shape == (num_boxes, 4)
        gt_fstarts = gt_anno["fstarts"]   # (n_gt,)
        gt_labels = gt_anno["labels"]     # (n_gt,)
        n_gt = len(gt_labels)

        viou_matrix = vIoU_broadcast(det_trajs,gt_trajs,det_fstarts,gt_fstarts)  # (n_det, n_gt)

        try:
            max_vious, gt_ids = torch.max(viou_matrix,dim=-1)  # shape == (n_det,)
        except:
            print(seg_tag,det_info)
            print(det_trajs)
            print(viou_matrix.shape)
        mask = max_vious > self.vIoU_th
        gt_ids[~mask] = n_gt
        gt_labels_with_bg = torch.constant_pad_nd(gt_labels,pad=(0,1),value=0) # (n_gt+1,)
        assigned_labels = gt_labels_with_bg[gt_ids]  # (n_det,)  range: 0~50, i.e., 0~base, 0 refers to __background__
        
        return deepcopy(seg_tag), assigned_labels


############ dataset for relation classification


class VidORGTDatasetForTrain(object):
    def __init__(self,
        dataset_dir = "/home/gkf/project/VidVRD_VidOR/vidor-dataset",
        cache_dir = "datasets/cache_vidor",
        tracking_res_dir = "/home/gkf/project/VidVRD-II/tracklets_results/VidORtrainVideoLevel_tracking_results_gt_th-15-5",
        traj_features_dir = "/home/gkf/project/scene_graph_benchmark/output/VidORtrain_gt_traj_features_th-15-5",
        traj_embds_dir = "/home/gkf/project/ALPRO/extract_features_output/VidOR_TrajFeatures256_gt_th-15-5",
        traj_cls_split_info_path = "configs/VidOR_OjbectClass_spilt_info_v2.json",
        pred_cls_split_info_path = "configs/VidOR_PredClass_spilt_info_v2.json",
        pred_cls_splits = ("base",),
        traj_cls_splits = ("base",),
        num_sample_segs = 48,   # this is deprecated, we directly use avg_num_seg as num_sample_segs
        remove_head_pred = True
    ):
        self.SEG_LEN = 30
        
        self.dataset_dir = dataset_dir
        self.cache_dir = cache_dir
        self.tracking_res_dir = tracking_res_dir
        self.traj_features_dir = traj_features_dir
        self.traj_embds_dir = traj_embds_dir
        traj_cls_split_info = load_json(traj_cls_split_info_path)
        self.traj_cls2id_map = traj_cls_split_info["cls2id"]
        self.traj_cls2split_map = traj_cls_split_info["cls2split"]
        pred_cls_split_info = load_json(pred_cls_split_info_path)
        self.pred_cls2id_map = pred_cls_split_info["cls2id"]
        self.pred_cls2split_map = pred_cls_split_info["cls2split"]
        self.pred_cls_splits = pred_cls_splits
        self.traj_cls_splits = traj_cls_splits
        self.remove_head_pred = remove_head_pred

        # self.num_sample_segs = num_sample_segs
        self.num_base = sum([v=="base" for v in pred_cls_split_info["cls2split"].values()])
        self.num_novel = sum([v=="novel" for v in pred_cls_split_info["cls2split"].values()])
        assert self.num_base == 30 and self.num_novel == 20

        segment_tags = self.prepare_segment_tags()
        
            
        LOGGER.info("loading gt_triplets and filter out segments with traj-split:{};pred-split:{} ...".format(self.traj_cls_splits,self.pred_cls_splits))
        self.gt_triplets,_ = self.get_gt_triplets(segment_tags)  # filter out segs that have no relation annotation
        labeled_segment_tags = sorted(self.gt_triplets.keys())
        # LOGGER.info("seg_tags:{}, labeled_seg_tags:{}".format(len(self.segment_tags),len(labeled_segment_tags)))

        self.video2seg, self.video_names = self.reset_video2segs(labeled_segment_tags)

        
        

        LOGGER.info("------------- dataset constructed -----------, len(self) == {}".format(len(self)))


    def prepare_segment_tags(self):
        self.anno_dir = os.path.join(self.dataset_dir,"annotation","training") # .../vidor-dataset/training
        group_ids = os.listdir(self.anno_dir)
        video_names = []
        for gid in group_ids:
            filenames = os.listdir(os.path.join(self.anno_dir,gid))
            video_names  += [gid + "_" + filename.split(".")[0]  for filename in filenames]

        video_names = sorted(video_names)
        
        cache_path = os.path.join(self.cache_dir,"VidORtrain_gtsupp_video2seg.json")
        if os.path.exists(cache_path):
            tmp = load_json(cache_path)
            video2seg = tmp["video2seg"]
            video2ntrajs = tmp["video2ntrajs"]
        else:
            video2seg = dict()
            video2ntrajs = dict()
            for video_name in tqdm(video_names,desc="prepare segment tags"):
                path = os.path.join(self.tracking_res_dir,video_name + ".json")
                track_res = load_json(path)
                # track_res = {seg_tag1:res_1,seg_tag2:res_2,...,"n_trajs":[16,15,24,0,14,...]}
                video2ntrajs[video_name] = track_res.pop("n_trajs")
                video2seg[video_name] = sorted(list(track_res.keys()))
            to_save = {"video2seg":video2seg,"video2ntrajs":video2ntrajs}
            with open(cache_path,'w') as f:
                json.dump(to_save,f)
        
        seg2relidx = dict()
        segment_tags = []
        for video_name in video_names:
            seg_tags = video2seg[video_name]
            segment_tags += seg_tags
            for idx,seg_tag in enumerate(seg_tags):
                seg2relidx[seg_tag] = idx
                
        self.seg2relidx = seg2relidx

        if self.remove_head_pred:
            ### remove video_names which only has head pred classes
            LOGGER.info("remove videos which only has head pred classes")
            video_names_ = _remove_HeadBasePred_video_names(self.pred_cls2split_map)
            segment_tags_ = []
            for seg_tag in segment_tags:
                video_name = seg_tag.split('-')[0]
                if video_name in video_names_:
                    segment_tags_.append(seg_tag)
            segment_tags = segment_tags_

        return segment_tags
    
    
    def reset_video2segs(self,labeled_segment_tags):
        video2seg_ = defaultdict(list)
        for seg_tag in labeled_segment_tags:
            video_name,fstart,fend = seg_tag.split('-') # e.g., 1010_8872539414-0825-0855
            video2seg_[video_name].append(seg_tag)

        video_names = sorted(list(video2seg_.keys()))

        n_segs = []
        for video_name,segs_ in video2seg_.items():
            n_segs.append(len(segs_))
        avg_ = sum(n_segs)/len(n_segs)
        LOGGER.info("num_segs per video: avg:{:.4f},max:{},min:{}, set num_sample_segs as avg".format(avg_,max(n_segs),min(n_segs)))

        self.num_sample_segs = int(avg_)
        
        return video2seg_,video_names
        


    def _get_cache_tag(self):
        # self.traj_cls_splits = ("base","novel")
        # self.pred_cls_splits = ("base",novel)
        
        traj_tag = [x[0].upper() for x in self.traj_cls_splits]
        traj_tag = "".join(traj_tag)
        pred_tag = [x[0].upper() for x in self.pred_cls_splits]
        pred_tag = "".join(pred_tag)

        cache_tag = "Traj_{}-Pred_{}".format(traj_tag,pred_tag)  # e.g., Traj_BN-Pred_N
        return cache_tag

    


    def get_gt_triplets(self,segment_tags):
        
        anno_dir = os.path.join(self.dataset_dir,"annotation","training") # .../vidor-dataset/training
        group_ids = sorted(os.listdir(anno_dir))
        file_paths = []
        for gid in group_ids:
            filenames = sorted(os.listdir(os.path.join(anno_dir,gid)))
            file_paths += [os.path.join(anno_dir,gid,name) for name in filenames]
        
        # avoid loading the same video's annotations multiple times
        video_annos = dict()
        for file_path in tqdm(file_paths,desc="loading annotations"):  # loop for 7000
            tmp = file_path.split('/') # .../vidor-dataset/annotation/training/0001/3058613626.json
            gid,vid = tmp[-2],tmp[-1].split('.')[0]
            video_name = gid + "_" + vid
            
            with open(file_path,'r') as f:
                anno_per_video = json.load(f)  # refer to `datasets/vidvrd-dataset/format.py`
            video_annos[video_name] = anno_per_video
        

        gt_triplets_all = dict()
        for seg_tag in tqdm(segment_tags,desc="segment-level filtering"):
            video_name, seg_fs, seg_fe = seg_tag.split('-')  # e.g., "1010_8872539414-0825-0855"
            seg_fs,seg_fe = int(seg_fs),int(seg_fe)
            anno = video_annos[video_name]
            trajid2cls_map = {traj["tid"]:traj["category"] for traj in anno["subject/objects"]}
            # e.g., {0: 'bicycle', 1: 'bicycle', 2: 'person', 3: 'person', 5: 'person'} # not necessarily continuous

            gt_triplets = defaultdict(list)
            relations = anno["relation_instances"]
            for rel in relations: # loop for relations of all segments in this video, some segments may have no annotation
                s_tid,o_tid = rel["subject_tid"],rel["object_tid"]  # tid w.r.t video-level
                s_cls = trajid2cls_map[s_tid]
                o_cls = trajid2cls_map[o_tid]
                pred_cls = rel["predicate"]   # pred class name
                if not (self.pred_cls2split_map[pred_cls] in self.pred_cls_splits):
                    continue
                if (self.traj_cls2split_map[s_cls] not in self.traj_cls_splits) or (self.traj_cls2split_map[o_cls] not in self.traj_cls_splits):
                    continue
                
                fs,fe =  rel["begin_fid"],rel["end_fid"]  # [fs,fe)  fe is exclusive (from annotation)
                if temporal_overlap((fs,fe),(seg_fs,seg_fe)) < self.SEG_LEN/2:
                    continue
                
                gt_triplets[(s_tid,o_tid)].append((s_cls,pred_cls,o_cls))
                # gt_triplets[(s_tid,o_tid)].append([(s_cls,pred_cls,o_cls),(fs,fe),(seg_fs,seg_fe)])   # for debug
            if len(gt_triplets) > 0:
                gt_triplets_all[seg_tag] = gt_triplets
        LOGGER.info("Done. {} segments left".format(len(gt_triplets_all)))

        return gt_triplets_all,video_annos


    def get_traj_infos(self,video_name,segment_tags):
        # we have filter originally saved tracking results,
        # refer to func:`filter_track_res_and_feature` in `/home/gkf/project/VidVRD-II/video_object_detection/object_bbox2traj_segment.py`
        '''
        tracking_results = {seg_tag1:res_1,seg_tag2:res_2,...,"n_trajs":[16,15,24,0,14,...]}
        res_i = [
            {
                'fstarts': int(fstart),     # relative frame idx w.r.t this segment
                'score': score,             # -1 for gt
                'bboxes': bboxes.tolist()   # list[list], len == num_frames, format xyxy

                ### for det traj
                'label':  cls_id            # int, w.r.t VinVL classes
                
                #### for gt_traj in train-set
                'class':  class_name        # str    
                'tid':tid                   # int, original tid in annotation (w.r.t video range)
                
            },
            ...
        ]  # len(res_i) == num_tracklets
        '''
        
        path = os.path.join(self.tracking_res_dir,video_name+".json")
        with open(path,'r') as f:
            tracking_results = json.load(f)

        rel_ids = [self.seg2relidx[seg_tag] for seg_tag in segment_tags]
        n_trajs_b4sp = tracking_results["n_trajs"]  # `b4sp` means before sample
        n_trajs = [n_trajs_b4sp[idx] for idx in rel_ids]

        traj_infos = []
        for seg_tag in segment_tags:
            fstarts = []
            scores = []
            bboxes = []
            clsids = []
            tids = []
            for res in tracking_results[seg_tag]:  # this loop can be empty
                fstarts.append(res["fstart"])  # relative frame_id  w.r.t segment fstart
                scores.append(res["score"])
                bboxes.append(torch.as_tensor(res["bboxes"])) # (num_boxes, 4)
                clsids.append(self.traj_cls2id_map[res["class"]]) #
                tids.append(res["tid"])
            
            traj_info = {
                "fstarts":torch.as_tensor(fstarts), # shape == (num_traj,)
                "scores":torch.as_tensor(scores),  # shape == (num_traj,)
                "bboxes":bboxes,  # list[tensor] , len== num_traj, each shape == (num_boxes, 4)
                "tids":torch.as_tensor(tids),        # shape == (num_traj,)
                "clsids":torch.as_tensor(clsids)    # object category id
            }
            traj_infos.append(traj_info)



        return traj_infos,n_trajs,n_trajs_b4sp



    def get_triplet_labels_BUG(self,gt_triplets,traj_tids):
        ##### NOTE BUG this func has KeyError BUG, refer to func:`get_triplet_labels` for correct implementation
        ## this `gt_triplets` is generated by func:`get_gt_triplets_deprecated`

        ### gt_triplets.keys() 中包括的 tid 要比 traj_tids 范围广
        # 所以不能把 traj_tids 中 的tid 作为 key，因为traj_tids 是在 traj_len_th 过滤之后的
        # gt_triplets 在构建的时候虽然是用了traj_len_th过滤之后的segment_tags来循环，但是在每个
        # seg_tag 对应的  frame 范围取到的relation的tid是video-level的，这个tid对应的轨迹可能跨越多个seg但是在当前seg中因为太短而被过滤掉了

        # 即，当前的seg_tag 对应的同一个 gt_triplets 和 traj_info["tids"] 中， gt_triplets.keys()的tid 可
        # 能不在 traj_info["tids"]中，因为gt_triplets.keys()的tid 是video-level的， 这个tid对应的轨迹
        # 可能跨越了多个segment, 但是在当前的seg_tag只有一小段，然后因为traj_len_th被过滤掉了
        ###  在VidVRD 中没有这个问题。因为小数据集VidVRD是按照segment来标注的，每个segment有标注的 traj都是跨越一整个seg的

        ## 所以正确的做法是， 将traj_info["tids"]先两两组合，然后去gt_triplets.keys() 中取，看看有没有。
        ## gt_triplets.keys() 未必包含 traj_info["tids"] 中所有两两组合的情况，同时 gt_triplets.keys() 中也存在这 不会被取到的key

        tid2idx_map = {tid:idx for idx,tid in enumerate(traj_tids.tolist())}
        
        pair_ids = []
        triplet_cls_ids = []
        for k,spo_list in gt_triplets.items():  # loop for n_pair
            s_tid,o_tid = k
            s_idx,o_idx = tid2idx_map[s_tid],tid2idx_map[o_tid]
            pair_ids.append([s_idx,o_idx])
            # triplet_cls_list.append(spo)
            spo_list = [
                [self.traj_cls2id_map[spo[0]],self.pred_cls2id_map[spo[1]],self.traj_cls2id_map[spo[2]]]
                for spo in spo_list
            ]  # len == n_preds
            triplet_cls_ids.append(torch.as_tensor(spo_list))  # (n_preds,3)
        pair_ids = torch.as_tensor(pair_ids)  # (n_pair,2)
        
        # LOGGER.info(list(gt_triplets.keys()),pair_ids.shape)
        # triplet_cls_ids len == n_pair each shape == (n_preds,3)

        return pair_ids,triplet_cls_ids
    

    def get_triplet_labels(self,gt_triplets,traj_tids):

        tid2idx_map = {tid:idx for idx,tid in enumerate(traj_tids.tolist())}
        psb_pair_tids = torch.cartesian_prod(traj_tids,traj_tids)  # prefix `psb` means possible (all possible pair tids)
        psb_pair_tids = set(tuple(pair) for pair in psb_pair_tids.tolist())
  
        pair_tids = psb_pair_tids & set(gt_triplets.keys())
        pair_tids = list(pair_tids)
        
        pair_ids = []
        triplet_cls_ids = []
        for so in pair_tids:  # loop for n_pair
            spo_list = gt_triplets[so]
            s_tid,o_tid = so

            s_idx,o_idx = tid2idx_map[s_tid],tid2idx_map[o_tid]
            pair_ids.append([s_idx,o_idx])            
            spo_list = [
                [self.traj_cls2id_map[spo[0]],self.pred_cls2id_map[spo[1]],self.traj_cls2id_map[spo[2]]]
                for spo in spo_list
            ]  # len == n_preds
            triplet_cls_ids.append(torch.as_tensor(spo_list))  # (n_preds,3)
        pair_ids = torch.as_tensor(pair_ids)  # (n_pair,2)
        
        # LOGGER.info(list(gt_triplets.keys()),pair_ids.shape)
        # triplet_cls_ids len == n_pair each shape == (n_preds,3)

        return pair_ids,triplet_cls_ids
    


    def get_pred_labels(self,gt_triplets,traj_tids):
        tid2idx_map = {tid:idx for idx,tid in enumerate(traj_tids.tolist())}
        psb_pair_tids = torch.cartesian_prod(traj_tids,traj_tids)  # prefix `psb` means possible (all possible pair tids)
        psb_pair_tids = set(tuple(pair) for pair in psb_pair_tids.tolist())
  
        pair_tids = psb_pair_tids & set(gt_triplets.keys())
        pair_tids = list(pair_tids)
        
        pair_ids = []
        n_pair = len(pair_tids)
        multihot = torch.zeros(size=(n_pair,1+self.num_base+self.num_novel))
        for i,so in enumerate(pair_tids):  # loop for n_pair
            s_tid,o_tid = so
            s_idx,o_idx = tid2idx_map[s_tid],tid2idx_map[o_tid]
            
            pair_ids.append([s_idx,o_idx])

            spo_list = gt_triplets[so]
            pred_clsids = [self.pred_cls2id_map[spo[1]] for spo in spo_list]
            pred_clsids = torch.as_tensor(pred_clsids) # (n_preds,)
            multihot[i,pred_clsids] = 1

        pair_ids = torch.as_tensor(pair_ids)  # (n_pair,2)
        return pair_ids, multihot
    
    def __len__(self):
        return len(self.video_names)


    def __getitem__(self, idx):
        video_name = deepcopy(self.video_names[idx])  # return video_name for debug
        segment_tags  = self.video2seg[video_name]

        num_sample_segs = min(self.num_sample_segs,len(segment_tags))
        segment_tags = np.random.choice(segment_tags,num_sample_segs,replace=False)

        traj_infos,n_trajs,n_trajs_b4sp = self.get_traj_infos(video_name,segment_tags)
        traj_features,traj_embds = VidORTrajDataset.get_traj_features(self,video_name,segment_tags,n_trajs_b4sp)
        
        assert len(n_trajs) == len(traj_features) and len(n_trajs) == len(traj_embds)
        for nt,tf,te in zip(n_trajs,traj_features,traj_embds):
            assert tf.shape[0] == te.shape[0] and nt == te.shape[0]

        
        vlv_s_feats = []        # `vlv` means video-level
        vlv_o_feats = []
        vlv_s_embds = []
        vlv_o_embds = []
        vlv_relpos_feats = []
        vlv_triplet_clsids = []  
        vlv_gt_pred_vecs = []
        for i,seg_tag in enumerate(segment_tags):
            traj_info = traj_infos[i]
            # pair_ids,triplet_cls_ids = self.get_triplet_labels(self.gt_triplets[seg_tag],traj_info["tids"])
            pair_ids,gt_pred_vecs = self.get_pred_labels(self.gt_triplets[seg_tag],traj_info["tids"])

            if pair_ids.numel() == 0:
                continue

            sids = pair_ids[:,0]  # (n_pair,)
            oids = pair_ids[:,1]

            
            s_feats = traj_features[i][sids,:]     # (n_pair,2048)
            o_feats = traj_features[i][oids,:]     # as above
            s_embds = traj_embds[i][sids,:]  # (n_pair,256)
            o_embds = traj_embds[i][oids,:]
            relpos_feats = get_relative_position_feature(traj_info["fstarts"],traj_info["bboxes"],sids,oids)  # (n_pair,12)

            
            vlv_s_feats.append(s_feats)
            vlv_o_feats.append(o_feats)
            vlv_s_embds.append(s_embds)
            vlv_o_embds.append(o_embds)
            vlv_relpos_feats.append(relpos_feats)
            # vlv_triplet_clsids += triplet_cls_ids
            vlv_gt_pred_vecs.append(gt_pred_vecs)
        
        if len(vlv_s_feats) == 0:
            rand_idx = np.random.choice(list(range(len(self))))
            LOGGER.info("video: {} with num_sample_segs={}, has no required data".format(video_name,num_sample_segs))
            return self.__getitem__(rand_idx)
        (
            vlv_s_feats,vlv_o_feats,vlv_s_embds,vlv_o_embds,vlv_relpos_feats,vlv_gt_pred_vecs
        ) = map(torch.cat,[vlv_s_feats,vlv_o_feats,vlv_s_embds,vlv_o_embds,vlv_relpos_feats,vlv_gt_pred_vecs]) 
        # default cat at dim=0, each shape == (N_pair, *)
        # return_items = (video_name,) + return_items + (vlv_triplet_clsids,)
        
        return_items = (
            video_name,
            vlv_s_feats,  # (N_pair, 2048), N_pair is total pairs in all segments of this video
            vlv_o_feats,
            vlv_s_embds,
            vlv_o_embds,
            vlv_relpos_feats,
            # vlv_triplet_clsids
            vlv_gt_pred_vecs
        )
        
        # video_name,s_roi_feats,o_roi_feats,s_embds,o_embds,relpos_feats,triplet_cls_ids
        return return_items
        

    def get_collator_func(self):

        # def default_collate_func(batch_data):
        #     # this collator_func simply swaps the order of inner and outer of batch_data

        #     bsz = len(batch_data)
        #     num_rt_vals = len(batch_data[0])

        #     return_values = tuple([batch_data[bid][vid] for bid in range(bsz)] for vid in range(num_rt_vals))
        #     return return_values
        
        def collate_func(batch_data):
            # video_names = vlv_s_feats,vlv_o_feats,vlv_s_embds,vlv_o_embds,vlv_relpos_feats,vlv_gt_pred_vecs = batch_data

            bsz = len(batch_data)
            num_rt_vals = len(batch_data[0])

            video_names = [batch_data[bid][0] for bid in range(bsz)]
            tensor_values = tuple(
                torch.cat([batch_data[bid][vid] for bid in range(bsz)],dim=0)
                for vid in range(1,num_rt_vals)
            )

            return_values = (video_names,) + tensor_values
            return return_values


        return collate_func


class VidORUnifiedDataset(object):
    def __init__(self,
        dataset_split,
        dataset_dir = "/home/gkf/project/VidVRD_VidOR/vidor-dataset",
        cache_dir = "datasets/cache_vidor",
        tracking_res_dir = {
            "train":"/home/gkf/project/VidVRD-II/tracklets_results/VidORtrainVideoLevel_tracking_results_th-15-5",
            "val":"/home/gkf/project/VidVRD-II/tracklets_results/VidORvalVideoLevel_tracking_results_th-15-5"
        },
        traj_features_dir={
            "train":"/home/gkf/project/scene_graph_benchmark/output/VidORtrain_traj_features_th-15-5",
            "val":"/home/gkf/project/scene_graph_benchmark/output/VidORval_traj_features_th-15-5",
        },
        traj_embds_dir = {
            "train":"/home/gkf/project/ALPRO/extract_features_output/VidOR_TrajFeatures256_th-15-5",
            "val":"/home/gkf/project/ALPRO/extract_features_output/VidORval_TrajFeatures256",
        },
        gt_training_traj_supp = dict(
            traj_dir = "/home/gkf/project/VidVRD-II/tracklets_results/VidORtrainVideoLevel_tracking_results_gt_th-15-5",
            feature_dir = "/home/gkf/project/scene_graph_benchmark/output/VidORtrain_gt_traj_features_th-15-5",
            embd_dir = "/home/gkf/project/ALPRO/extract_features_output/VidOR_TrajFeatures256_gt_th-15-5",
        ), # only used for train
        traj_cls_split_info_path = "configs/VidOR_OjbectClass_spilt_info_v2.json",
        pred_cls_split_info_path = "configs/VidOR_PredClass_spilt_info_v2.json",
        pred_cls_splits = ("base",), # only used for train
        traj_cls_splits = ("base",), # only used for train
        subset_idx_range = [0,7000], # only used for train
        num_sample_segs = 58,        # only used for train
        num_sample_pairs = 4096,     # only used for train
        vpoi_th = 0.9,               # only used for train
        pos_ratio = 0.75,            # only used for train
        for_assign_label = False,    # only used for train
        remove_head_pred = True,     # only used for train
        label_cache_dir = "prepared_data/vidor_AssignedPredLabels_Traj_B-Pred_B_vpoi-0.90"
    ):
        self.SEG_LEN = 30
        self.dataset_split = dataset_split.lower() # "train" or "val"
        self.traj_cls_splits = traj_cls_splits # we do not assign traj labels here, but the triplets contain trajs that need to be specified `base` or `novel`
        self.pred_cls_splits  = pred_cls_splits
        
        traj_cls_split_info = load_json(traj_cls_split_info_path)
        self.traj_cls2id_map = traj_cls_split_info["cls2id"]
        self.traj_cls2split_map = traj_cls_split_info["cls2split"]
        pred_cls_split_info = load_json(pred_cls_split_info_path)
        self.pred_cls2id_map = pred_cls_split_info["cls2id"]
        self.pred_cls2split_map = pred_cls_split_info["cls2split"]
        self.pred_num_base = sum([v=="base" for v in pred_cls_split_info["cls2split"].values()])
        self.pred_num_novel = sum([v=="novel" for v in pred_cls_split_info["cls2split"].values()])
        assert self.pred_num_base == 30 and self.pred_num_novel == 20
        self.num_sample_segs = num_sample_segs
        self.num_sample_pairs = num_sample_pairs
        self.vpoi_th = vpoi_th
        self.pos_ratio = pos_ratio
        self.remove_head_pred = remove_head_pred
        

        self.dataset_dir = dataset_dir 
        self.cache_dir = cache_dir
        self.gt_training_traj_supp = gt_training_traj_supp
        self.tracking_res_dir = tracking_res_dir[self.dataset_split]
        # .../tracklets_results/VidORtrain_tracking_results_th-15-5/1010_8872539414/1010_8872539414-0825-0855.json
        self.traj_features_dir = traj_features_dir[self.dataset_split]
        self.traj_embds_dir = traj_embds_dir[self.dataset_split]
        
        
        segment_tags_all = self.prepare_segment_tags()  #
        self.segment_tags_all = segment_tags_all
        
        
        if self.gt_training_traj_supp is not None:
            assert self.dataset_split == "train"
            supp_tag = "_with_GTsupp"
        else:
            supp_tag = ""

        self.for_assign_label = for_assign_label
        self.label_cache_path = os.path.join(cache_dir,"VidORtrain_PredLabels{}_th-{}.pth".format(supp_tag,vpoi_th))
        self.label_cache_dir = label_cache_dir
        ####### TODO modify the name of label_cache_dir (add supp_tag & vpoi_th & Traj-Pred cls_split)

        if for_assign_label:
            assert self.dataset_split == "train"
            LOGGER.info("loading gt_triplets and filter out segments with traj-split:{};pred-split:{} ...".format(self.traj_cls_splits,self.pred_cls_splits))
            self.gt_triplets,self.video_annos = VidORGTDatasetForTrain.get_gt_triplets(self,segment_tags_all)
            labeled_segment_tags = sorted(self.gt_triplets.keys())
            self.video2seg, video_names = VidORGTDatasetForTrain.reset_video2segs(self,labeled_segment_tags)
            s,e = subset_idx_range
            self.video_names = video_names[s:e]
            return
        
        if self.dataset_split == "val":
            self.val_vid2videoname_map = {video_name.split('_')[1]:video_name for video_name in self.video_names}


        if self.dataset_split == "train": #  "we only do label assignment for train set"     
            split_tag = VidORGTDatasetForTrain._get_cache_tag(self)  # e.g., Traj_B-Pred_B
            labeled_video_names = [x.split('.')[0] for x in  os.listdir(self.label_cache_dir)]
            labeled_video_names = sorted(labeled_video_names)
            
            video2seg, labeled_video_names_ = self._get_labeled_video2seg_cache(
                "VidORtrain{}_{}_labeled_video2seg.json".format(supp_tag,split_tag)
            )
            # labeled_video_names_ is not used, we use labeled_video_names from label_cache_dir
            # assert sorted(labeled_video_names_) == labeled_video_names ## TODO assert this after all label_assignment done.
            
            LOGGER.info("len(labeled_video_names) = {}".format(len(labeled_video_names)))
            
            self.video2seg = video2seg  
            self.video_names = labeled_video_names

            if self.remove_head_pred:
                ### remove video_names which only has head pred classes
                LOGGER.info("remove videos which only has head pred classes")
                video_names_ = _remove_HeadBasePred_video_names(self.pred_cls2split_map)
                video_names_ = set(self.video_names) & set(video_names_)
                self.video_names = sorted(video_names_)
                
                # NOTE video2seg is a dict, leave it without change is fine

        LOGGER.info("--------------- dataset constructed  len(self) == {}---------------".format(len(self)))
    
    def __len__(self):
        
        # if self.dataset_split == "train":
        #     return len(self.video_names)
        # else:
        #     return len(self.segment_tags_all)
        return len(self.video_names)

    def _get_labeled_video2seg_cache(self,filename):
        ## e.g., filename == "VidORtrain_gtsupp_{split_tag}_labeled_video2seg.json"
        

        cache_path = os.path.join(self.cache_dir,filename)
        if os.path.exists(cache_path):
            tmp = load_json(cache_path)
            video2seg = tmp["video2seg"]
            video_names = tmp["video_names"]
        else:
            LOGGER.info("prepare labeled_video2seg and save as cache ...")
            gt_triplets,_ = VidORGTDatasetForTrain.get_gt_triplets(self,self.segment_tags_all)
            labeled_segment_tags = sorted(gt_triplets.keys())
            video2seg, video_names = VidORGTDatasetForTrain.reset_video2segs(self,labeled_segment_tags)

            to_save = {"video2seg":video2seg,"video_names":video_names}
            LOGGER.info("saving labeled_video2seg at cache_path:{}".format(cache_path))
            with open(cache_path,'w') as f:
                json.dump(to_save,f) 
        
        return video2seg,video_names


    def _get_video2seg_cache(self,filename,tracking_res_dir):
        ## e.g., filename == "VidORtrain_gtsupp_video2seg.json"

        cache_path = os.path.join(self.cache_dir,filename)
        if os.path.exists(cache_path):
            tmp = load_json(cache_path)
            video2seg = tmp["video2seg"]
            video2ntrajs = tmp["video2ntrajs"]
        else:
            video2seg = dict()
            video2ntrajs = dict()
            for video_name in tqdm(self.video_names_all,desc="prepare segment tags",ncols=160):
                path = os.path.join(tracking_res_dir,video_name + ".json")
                track_res = load_json(path)
                # track_res = {seg_tag1:res_1,seg_tag2:res_2,...,"n_trajs":[16,15,24,0,14,...]}
                video2ntrajs[video_name] = track_res.pop("n_trajs")
                video2seg[video_name] = sorted(list(track_res.keys()))
            to_save = {"video2seg":video2seg,"video2ntrajs":video2ntrajs}
            LOGGER.info("saving segment_tags at cache_path:{}".format(cache_path))
            with open(cache_path,'w') as f:
                json.dump(to_save,f) 
        
        return video2seg,video2ntrajs

    def prepare_segment_tags(self):
        self.anno_dir = os.path.join(self.dataset_dir,"annotation",_reset_dataset_split(self.dataset_split)) # .../vidor-dataset/training
        group_ids = os.listdir(self.anno_dir)
        video_names_all = []
        for gid in group_ids:
            filenames = os.listdir(os.path.join(self.anno_dir,gid))
            video_names_all  += [gid + "_" + filename.split(".")[0]  for filename in filenames]

        video_names_all = sorted(video_names_all)
        self.video_names_all = video_names_all

        video2seg,video2trajs = self._get_video2seg_cache(
            "VidOR{}_det_video2seg.json".format(self.dataset_split),
            self.tracking_res_dir
        )

        seg2relidx = dict()
        segment_tags_all = []
        for video_name, seg_tags in video2seg.items():
            segment_tags_all += seg_tags
            for rel_idx,seg_tag in enumerate(seg_tags):
                seg2relidx[seg_tag] = rel_idx

        self.seg2relidx = seg2relidx
        segment_tags_all = sorted(segment_tags_all)
        
        self.video2seg = video2seg  # this is for val-set, for train-set, video2seg & video_names will be reset
        self.video_names = video_names_all

        if self.dataset_split=="train" and (self.gt_training_traj_supp is not None):        

            video2seg_gt,video2trajs_gt = self._get_video2seg_cache(
                "VidORtrain_gtsupp_video2seg.json",
                self.gt_training_traj_supp["traj_dir"]
            )
            seg2relidx_gt = dict()
            for video_name, seg_tags in video2seg_gt.items():
                for rel_idx,seg_tag in enumerate(seg_tags):
                    seg2relidx_gt[seg_tag] = rel_idx

            self.video2seg_gt = video2seg_gt
            self.seg2relidx_gt = seg2relidx_gt

        return segment_tags_all

    def set_gt_suppfile_info(self):
        # this func is deprecated
        # this is used for func:`VidORTrajDataset.get_traj_infos` & `VidORTrajDataset.get_traj_features` 

        gt_suppfile_info = {
            "tracking_res_dir":self.gt_training_traj_supp["traj_dir"],
            "traj_features_dir":self.gt_training_traj_supp["feature_dir"],
            "traj_embds_dir":self.gt_training_traj_supp["embd_dir"],
            "seg2relidx": self.seg2relidx_gt
        }
        self.gt_suppfile_info = EasyDict(gt_suppfile_info)
    
    def set_det_trajfile_info(self):
        # this func is deprecated
        # this is used for func:`VidORTrajDataset.get_traj_infos` & `VidORTrajDataset.get_traj_features` 

        det_trajfile_info = {
            "tracking_res_dir":self.tracking_res_dir,
            "traj_features_dir":self.traj_features_dir,
            "traj_embds_dir":self.traj_embds_dir,
            "seg2relidx": self.seg2relidx
        }
        self.det_trajfile_info = EasyDict(det_trajfile_info)

    
    def merge_gt_trajinfo(self,traj_infos,video_name,segment_tags):
        segment_tags_gt = self.video2seg_gt[video_name]

        path = os.path.join(self.gt_training_traj_supp["traj_dir"],video_name+".json")
        with open(path,'r') as f:
            tracking_results_gt = json.load(f)

        # rel_ids = [self.seg2relidx_gt[seg_tag] for seg_tag in segment_tags_gt]
        
        
        det2gt_ids = []
        for idx,seg_tag in enumerate(segment_tags):
            if seg_tag not in segment_tags_gt:
                continue
            
            det2gt_ids.append(
                (idx,self.seg2relidx_gt[seg_tag])
            )

            fstarts = []
            scores = []
            bboxes = []
            for res in tracking_results_gt[seg_tag]:  # this loop can be empty
                fstarts.append(res["fstart"])  # relative frame_id  w.r.t segment fstart
                scores.append(res["score"])  # for gt traj, score is -1  
                bboxes.append(torch.as_tensor(res["bboxes"])) # (num_boxes, 4)

            fstarts = torch.as_tensor(fstarts) # shape == (num_traj,) this can be empty, i.e., fstarts = torch.as_tensor([])
            scores = torch.as_tensor(scores)  # shape == (num_traj,)

            ### update
            traj_infos[idx]["fstarts"] =  torch.cat([traj_infos[idx]["fstarts"],fstarts],dim=0) # cat an empty-tensor is fine
            traj_infos[idx]["scores"] =  torch.cat([traj_infos[idx]["scores"],scores],dim=0)
            traj_infos[idx]["bboxes"] = traj_infos[idx]["bboxes"] + bboxes


        n_trajs_b4sp = tracking_results_gt.pop("n_trajs")  # `b4sp` means before sample
        n_trajs = [n_trajs_b4sp[gt_idx] for _,gt_idx in det2gt_ids] # if some idx in rel_ids_gt whose scores==[], then the corresponding n_trajs[idx]==0
        
        return traj_infos,det2gt_ids,n_trajs_b4sp
  
    def merge_gt_trajfeatures(self,traj_features,traj_embds,video_name,det2gt_ids,n_trajs):

        #### traj RoI features (2048-d)
        # /home/gkf/project/scene_graph_benchmark/output/VidORtrain_gt_traj_features/0000_2401075277_traj_features.npy
        path = os.path.join(self.gt_training_traj_supp["feature_dir"],video_name+"_traj_features.npy")
        traj_features_gt = np.load(path) 
        traj_features_gt = torch.from_numpy(traj_features_gt).float()  # float32, # (N_traj,2048)          
        traj_features_gt = torch.split(traj_features_gt,n_trajs,dim=0)  # len == N_seg (before sample)
        
        #### traj Alpro-embeddings (256-d)
        # /home/gkf/project/ALPRO/extract_features_output/VidOR_TrajFeatures256_gt/0000_2401075277.pth
        # (N_traj,258) format of each line: [seg_id,tid,256d-feature] (tid w.r.t segment range (before traj_len filter), not original anno)
        path = os.path.join(self.gt_training_traj_supp["embd_dir"],video_name+".pth")
        tmp = torch.load(path)  # (N_traj,258)
        seg_rel_ids = tmp[:,0]  # (N_traj,)
        tids_aft_filter = tmp[:,1]  #this is deprecated, because we apply traj_len_th filter directly after the Seq-NMS and save the tracking results .json file
        traj_embds_gt = tmp[:,2:]

        traj_embds_gt = torch.split(traj_embds_gt,n_trajs,dim=0)

        # det2gt_ids_debug = dict()
        for idx_det,idx_gt in det2gt_ids:
            # len_det = traj_features[idx_det].shape[0]
            # len_gt = traj_features_gt[idx_gt].shape[0]
            # det2gt_ids_debug[idx_det] = [len_det,len_gt]

            # idx_det w.r.t sampled segment_tags order, and traj_features are also w.r.t sampled order
            # idx_gt w.r.t original order of gt-segment_tags, and traj_features_gt w.r.t original order
            traj_features[idx_det] = torch.cat([traj_features[idx_det],traj_features_gt[idx_gt]],dim=0)
            traj_embds[idx_det] = torch.cat([traj_embds[idx_det],traj_embds_gt[idx_gt]],dim=0)

        # return traj_features,traj_embds,det2gt_ids_debug
        return traj_features,traj_embds

    
    def __getitem__(self, idx):
        if self.for_assign_label:
            return self._getitem_for_assign_label(idx)
        
        # if self.dataset_split == "val":  this is deprecated
        #     return self._gettiem_for_eval(idx)
        
        video_name = self.video_names[idx]  # return video_name for debug
        segment_tags  = self.video2seg[video_name]  # some segs_might not have label
        
        if self.dataset_split == "train":
            num_sample_segs = min(self.num_sample_segs,len(segment_tags))
            segment_tags = np.random.choice(segment_tags,num_sample_segs,replace=False).tolist()

        #### NOTE in get_traj_infos & get_traj_features, seg2relidx is w.r.t detection segs, 
        # (即使这里的segment_tags是从 labeled video2seg 中取出来的， 我们在get_traj_infos的时候还是去算seg2relidx is w.r.t detection segs，
        # 而不是 w.r.t labeled video2seg)
        #### And in assigned labels we must not use `self.seg2relidx`
        traj_infos,n_trajs,n_trajs_b4sp = VidORTrajDataset.get_traj_infos(self,video_name,segment_tags,rt_ntrajs_only=False)
        traj_features,traj_embds = VidORTrajDataset.get_traj_features(self,video_name,segment_tags,n_trajs_b4sp)
        
        # print(video_name,n_trajs,len(segment_tags),len(n_trajs),len(traj_features),len(traj_embds))
        if self.gt_training_traj_supp is not None:
            traj_infos,det2gt_ids,n_trajs_b4sp = self.merge_gt_trajinfo(traj_infos,video_name,segment_tags)
            # print(det2gt_ids,"-="*10)
            if det2gt_ids:   
                # i.e., if det2gt_ids not empty
                # traj_features,traj_embds,det2gt_ids_debug = self.merge_gt_trajfeatures(traj_features,traj_embds,video_name,det2gt_ids,n_trajs_b4sp)
                traj_features,traj_embds = self.merge_gt_trajfeatures(traj_features,traj_embds,video_name,det2gt_ids,n_trajs_b4sp)
        ##### for debug
        # for idx,seg_tag in enumerate(segment_tags):
        #     traj_info = traj_infos[idx]
        #     print(seg_tag,traj_info["scores"],det2gt_ids_debug[idx])


        vlv_relpos_feats = []
        vlv_s_feats = []
        vlv_o_feats = []
        vlv_s_embds = []
        vlv_o_embds = []
        for idx,seg_tag in enumerate(segment_tags):
            traj_info = traj_infos[idx]
            # n_traj = n_trajs[idx]
            n_traj = len(traj_info["bboxes"])
            # print(seg_tag,traj_features[idx].shape[0],n_traj,traj_embds[idx].shape[0],"--------------",det2gt_ids_debug[idx])
            assert traj_features[idx].shape[0] == n_traj and n_traj == traj_embds[idx].shape[0]
            
            
            pair_ids = trajid2pairid(n_traj)  # (n_pair,2)  n_pair == n_det*(n_det-1)
            sids = pair_ids[:,0]  # (n_pair,)
            oids = pair_ids[:,1]

            if pair_ids.numel() == 0:  # if n_traj==1, no pair
                relpos_feats = torch.empty(size=(0,12))
                # LOGGER.info("empty pair_ids,n_traj={},seg_rel_idx={},seg_tag={}".format(n_traj,idx,seg_tag))
            else:
                relpos_feats = get_relative_position_feature(traj_info["fstarts"],traj_info["bboxes"],sids,oids)  # (n_pair,12)
            vlv_relpos_feats.append(relpos_feats)

            if self.dataset_split == "train":
                vlv_s_feats.append(traj_features[idx][sids,:])
                vlv_o_feats.append(traj_features[idx][oids,:])
                vlv_s_embds.append(traj_embds[idx][sids,:])
                vlv_o_embds.append(traj_embds[idx][oids,:])
        
        
        if self.dataset_split == "train":
            vlv_labels,vlv_masks = self._get_pred_labels(video_name,segment_tags)  # each shape (n_pair,n_cls), n_cls = 1+n_base+n_novel
            vlv_labels = torch.cat(vlv_labels,dim=0) # (N_pos_pair,n_cls), N_pos_pair is total postive subj-obj pairs in all sampled segs in this video
            vlv_masks = torch.cat(vlv_masks,dim=0)  # (N_pair,) 
            if vlv_labels.numel() == 0:  # i.e., if vlv_masks.sum() == 0
                rand_idx = np.random.choice(list(range(len(self))))
                return self.__getitem__(rand_idx)
            
            (
                vlv_s_feats,vlv_o_feats,vlv_s_embds,vlv_o_embds,vlv_relpos_feats
            ) = map(torch.cat,[vlv_s_feats,vlv_o_feats,vlv_s_embds,vlv_o_embds,vlv_relpos_feats])
            assert vlv_masks.shape[0] == vlv_s_feats.shape[0]  # i.e., N_pair

            ### sample negative samples
            pos_mask = vlv_masks
            neg_mask = self.negative_sampling(pos_mask)
           
            # vlv_labels = torch.constant_pad_nd(vlv_labels,pad=(0,0,0,neg_mask.sum()))
            neg_labels = torch.zeros(size=(neg_mask.sum(),1+self.pred_num_base+self.pred_num_novel))
            neg_labels[:,0] = 1
            vlv_labels = torch.cat([vlv_labels,neg_labels],dim=0)
            cat_neg_func = lambda x : torch.cat([x[pos_mask,:],x[neg_mask,:]],dim=0)
            (
                vlv_s_feats,vlv_o_feats,vlv_s_embds,vlv_o_embds,vlv_relpos_feats
            ) = map(cat_neg_func,[vlv_s_feats,vlv_o_feats,vlv_s_embds,vlv_o_embds,vlv_relpos_feats])
            assert vlv_labels.shape[0] == vlv_s_feats.shape[0]  # i.e., N_pair_after_sample

            ### sample pairs
            (
                vlv_s_feats,vlv_o_feats,vlv_s_embds,vlv_o_embds,vlv_relpos_feats,vlv_labels
            ) = map(self.pair_sampling,[vlv_s_feats,vlv_o_feats,vlv_s_embds,vlv_o_embds,vlv_relpos_feats,vlv_labels])
            # print(vlv_s_feats.shape)
            rt_values = (
                video_name,
                vlv_s_feats,
                vlv_o_feats,
                vlv_s_embds,
                vlv_o_embds,
                vlv_relpos_feats,
                vlv_labels
            )
        else:
            rt_values  = (
                segment_tags,
                traj_infos,
                traj_features,
                traj_embds,
                vlv_relpos_feats
            )

        return rt_values
    
    def pair_sampling(self,tensor):
        num_sample = min(tensor.shape[0],self.num_sample_pairs)
        perm = torch.randperm(tensor.shape[0])
        idx = perm[:num_sample]
        tensor = tensor[idx,:]
        return tensor

    def negative_sampling(self,pos_mask):

        # refer to test_API/test_negative_sampling.py

        neg_mask = ~pos_mask
        num_pos = pos_mask.sum()
        num_neg = neg_mask.sum()
        
        # n_drop = num_neg - num_pos  # this can be < 0  <-- this is manually control 1:1 sample
        n_drop = (num_neg + num_pos) - int(num_pos/self.pos_ratio)

        if n_drop <= 0: 
            # when there are more pos-sample than neg-sample (n < p*(1-r)/r)
            return neg_mask
        
        idx = neg_mask.nonzero(as_tuple=True)[0]  # idx.shape == (n_neg,)
        perm = torch.randperm(num_neg)
        idx = idx[perm[:n_drop]] 
        # NOTE if n_drop < 0 , we must not use `idx[perm[:n_drop]]`, refer to refer to test_API/test_negative_sampling.py
        neg_mask[idx] = False

        return neg_mask

    def _get_pred_labels(self,video_name,segment_tags):

        ## refer to `prepared_data/cvt_pred_labels.py`

        labels_dict = torch.load(os.path.join(self.label_cache_dir,video_name+".pth"))
        labels = [labels_dict[seg_tag][0] for seg_tag in segment_tags]  # （n_pos_pair,n_cls)
        pos_masks = [labels_dict[seg_tag][1] for seg_tag in segment_tags]  # (n_pair,)

        
        return labels,pos_masks

        

    def _gettiem_for_eval(self,idx):
        ## this is deprecated
        ## 40 min for num_workers = 12, 相比之下， w.r.t video只要十分钟 with num_workers=8
        seg_tag = deepcopy(self.segment_tags_all[idx])
        video_name = seg_tag.split('-')[0]
        traj_infos,n_trajs,n_trajs_b4sp = VidORTrajDataset.get_traj_infos(self,video_name,[seg_tag],rt_ntrajs_only=False)
        traj_features,traj_embds = VidORTrajDataset.get_traj_features(self,video_name,[seg_tag],n_trajs_b4sp)

        traj_info = traj_infos[0]
        n_traj = n_trajs[0]
        traj_feature = traj_features[0]
        traj_embd  = traj_embds[0]

        assert n_traj == len(traj_info["bboxes"])
        pair_ids = trajid2pairid(n_traj)  # (n_pair,2)  n_pair == n_det*(n_det-1)
        sids = pair_ids[:,0]  # (n_pair,)
        oids = pair_ids[:,1]
        if pair_ids.numel() == 0:  # if n_traj==1, no pair
            relpos_feats = torch.empty(size=(0,12))
            # print(traj_embds[idx][sids,:].shape)
            # LOGGER.info("empty pair_ids,n_traj={},seg_rel_idx={},seg_tag={}".format(n_traj,idx,seg_tag))
        else:
            relpos_feats = get_relative_position_feature(traj_info["fstarts"],traj_info["bboxes"],sids,oids)  # (n_pair,12)
        
        return seg_tag,traj_info,traj_feature,traj_embd,relpos_feats
        


    def reset_video_namse_for_assign_label(self,video_names_undo):
        self.video_names = video_names_undo
        

    def _getitem_for_assign_label(self,idx):
        video_name = self.video_names[idx]  # return video_name for debug
        segment_tags  = self.video2seg[video_name]


        traj_infos,n_trajs,n_trajs_b4sp = VidORTrajDataset.get_traj_infos(self,video_name,segment_tags,rt_ntrajs_only=False)
        

        if self.gt_training_traj_supp is not None:
            traj_infos,det2gt_ids,n_trajs_b4sp = self.merge_gt_trajinfo(traj_infos,video_name,segment_tags)
        
        assert len(traj_infos) == len(segment_tags)
        labels_dict = dict()
        n_seg = len(segment_tags)
        for i in tqdm(range(n_seg),desc="assign label for video: {}".format(video_name),ncols=160,position=1,leave=False):
            traj_info,seg_tag = traj_infos[i],segment_tags[i]
            seg_tag,assigned_pred_labels,mask = self.assign_label(traj_info,seg_tag)
            # assigned_pred_labels.shape == (n_pair,1+num_base+num_novel)
            labels_dict[seg_tag] = assigned_pred_labels
        
        return labels_dict
        

    def get_gt_trajanno(self,seg_tag):
        video_name, fstart, fend = seg_tag.split('-')  # e.g., "ILSVRC2015_train_00010001-0016-0048"
        fstart,fend = int(fstart),int(fend)

        anno = self.video_annos[video_name]
        trajid2cls_map = {traj["tid"]:traj["category"] for traj in anno["subject/objects"]}

        trajs_info = {tid:defaultdict(list) for tid in trajid2cls_map.keys()}

        annotated_len = len(anno["trajectories"])
        
        for frame_id in range(fstart,fend,1):  # 75， 105
            if frame_id >= annotated_len:  
                # e.g., for "ILSVRC2015_train_00008005-0075-0105", annotated_len==90, and anno_per_video["frame_count"] == 130
                break

            frame_anno = anno["trajectories"][frame_id]  
            # NOTE frame_anno can be [] (empty) for all `fstart` to `fend`, because some segments have no annotation
            # print(seg_tag,frame_id,len(frame_anno))
            for bbox_anno in frame_anno:  
                tid = bbox_anno["tid"]
                bbox = bbox_anno["bbox"]
                bbox = [bbox["xmin"],bbox["ymin"],bbox["xmax"],bbox["ymax"]]
                trajs_info[tid]["bboxes"].append(bbox)
                trajs_info[tid]["frame_ids"].append(frame_id)

        frame_range_max_diff = 0
        trajs_info_ = dict()
        for tid, info in trajs_info.items():
            if not info:  # i.e., if `info` is empty, we continue
                # this can happen, because some segments have no annotation
                continue
            
            frame_range = max(info["frame_ids"]) - min(info["frame_ids"]) + 1
            # assert len(info["bboxes"]) == frame_range
            if frame_range - len(info["bboxes"]) > frame_range_max_diff:
                # frame_range_mismatch = True
                frame_range_max_diff = frame_range - len(info["bboxes"])
                # info_str = "seg_tag:{} len(frame_ids) != frame_range; frame_ids:{}".format(seg_tag,info["frame_ids"])
                # LOGGER.info(info_str)
                # [240, 241, 242, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269]
            ## refer to func:`frame_range_stat` in `tools/dataloader_demo_vidor.py`
            ## segment that len(info["bboxes"]) != frame_range is < 1%

            trajs_info_[tid] = {
                "class":trajid2cls_map[tid],
                "fstarts": min(info["frame_ids"]),  # relative frame_id  w.r.t the whole video
                "bboxes": torch.as_tensor(info["bboxes"])  # shape == (num_bbox,4)
            }
            
        return trajs_info_,frame_range_max_diff




    def assign_label(self, det_traj_info, seg_tag):
        
        video_name, seg_fs, seg_fe = seg_tag.split('-')  # e.g., "ILSVRC2015_train_00010001-0016-0048"
        seg_fs,seg_fe = int(seg_fs),int(seg_fe)
        gt_trajanno,_ = self.get_gt_trajanno(seg_tag)
        
        ## 1. construct gt traj pair
        gt_triplets = self.gt_triplets[seg_tag]
        n_gt_pair = len(gt_triplets)

        gt_s_trajs = []
        gt_o_trajs = []
        gt_s_fstarts = []
        gt_o_fstarts = []
        gt_pred_vecs = torch.zeros(size=(n_gt_pair,1+self.pred_num_base+self.pred_num_novel))  # (num_pred_cls,) n_base+n_novel+1, including background
        for i,(k,spo_cls_list) in enumerate(gt_triplets.items()): # loop for traj pair
            s_tid,o_tid = k
            p_clsnames = [spo_cls[1] for spo_cls in spo_cls_list]
            p_clsids  = [self.pred_cls2id_map[name] for name in p_clsnames]
            p_clsids = torch.as_tensor(p_clsids)
            gt_pred_vecs[i,p_clsids] = 1
            
            s_traj = gt_trajanno[s_tid]
            o_traj = gt_trajanno[o_tid]
            
            s_fs = s_traj["fstarts"]  # w.r.t the whole video
            o_fs = o_traj["fstarts"]
            s_boxes = s_traj["bboxes"]
            o_boxes = o_traj["bboxes"]
            
            gt_s_trajs.append(s_boxes)
            gt_o_trajs.append(o_boxes)
            gt_s_fstarts.append(s_fs - seg_fs) # w.r.t the segment
            gt_o_fstarts.append(o_fs - seg_fs) 
        
        gt_s_fstarts = torch.as_tensor(gt_s_fstarts)  # (n_gt_pair,)
        gt_o_fstarts = torch.as_tensor(gt_o_fstarts)

        ## 2. construct det traj pair

        det_trajs = det_traj_info["bboxes"]    # list[tensor] , len == n_det, each shape == (num_boxes, 4)
        det_fstarts = det_traj_info["fstarts"]  # (n_det,)
        if det_fstarts.dtype == torch.float32:
            det_fstarts = det_fstarts.type(torch.int64)
            
        pair_ids = trajid2pairid(len(det_trajs))  # (n_pair,2)  n_pair == n_det*(n_det-1)
        s_ids = pair_ids[:,0]  # (n_pair,)
        o_ids = pair_ids[:,1]

        det_s_fstarts = det_fstarts[s_ids]
        det_o_fstarts = det_fstarts[o_ids]
        det_s_trajs = [det_trajs[idx] for idx in s_ids]
        det_o_trajs = [det_trajs[idx] for idx in o_ids]

        ## 3. calculate vPoI and assign label
        # gt_s_trajs_len = [len(x) for x in gt_s_trajs]
        # print(gt_s_trajs_len,"gt_s_trajs_len")
        # print(seg_tag,det_s_fstarts)
        vpoi_s = vPoI_broadcast(det_s_trajs,gt_s_trajs,det_s_fstarts,gt_s_fstarts)  # (n_pair, n_gt_pair)  # 
        vpoi_o = vPoI_broadcast(det_o_trajs,gt_o_trajs,det_o_fstarts,gt_o_fstarts)  # (n_pair, n_gt_pair)
        vpoi_mat = torch.minimum(vpoi_s,vpoi_o)

        max_vpois,gt_pair_ids = torch.max(vpoi_mat,dim=-1)  # (n_pair,)
        # for each traj_pair, assign the gt_pair that has the max vPoI to it.
        mask = max_vpois > self.vpoi_th  # (n_pair, )
        assigned_pred_labels = gt_pred_vecs[gt_pair_ids,:]  # (n_pair,num_pred_cats)

        assigned_pred_labels[~mask,:] = 0  # first, set background target as all-zero vectors (overwrite other multihot vectors)
        assigned_pred_labels[~mask,0] = 1  # then set these all-zero vectors as [1,0,0,...,0]  

        ## TODO, only store positive labels & mask to save memory
        ## refer to `prepared_data/cvt_pred_labels.py`

        return seg_tag,assigned_pred_labels,mask

    
    def get_collator_func(self):

        # def default_collate_func(batch_data):
        #     # this collator_func simply swaps the order of inner and outer of batch_data

        #     bsz = len(batch_data)
        #     num_rt_vals = len(batch_data[0])

        #     return_values = tuple([batch_data[bid][vid] for bid in range(bsz)] for vid in range(num_rt_vals))
        #     return return_values
        if self.for_assign_label:
            return lambda x:x[0]
        
        def collate_func(batch_data):
            # video_names = vlv_s_feats,vlv_o_feats,vlv_s_embds,vlv_o_embds,vlv_relpos_feats,vlv_gt_pred_vecs = batch_data

            bsz = len(batch_data)
            num_rt_vals = len(batch_data[0])

            video_names = [batch_data[bid][0] for bid in range(bsz)]
            tensor_values = tuple(
                torch.cat([batch_data[bid][vid] for bid in range(bsz)],dim=0)
                for vid in range(1,num_rt_vals)
            )

            return_values = (video_names,) + tensor_values
            return return_values
        
        return collate_func


class VidORUnifiedDataset_ForEval(VidORUnifiedDataset):
    def __init__(self, 
        dataset_dir = "/home/gkf/project/VidVRD_VidOR/vidor-dataset",
        cache_dir = "datasets/cache_vidor",
        tracking_res_dir = "/home/gkf/project/VidVRD-II/tracklets_results/VidORvalVideoLevel_tracking_results_th-15-5",
        traj_features_dir= "/home/gkf/project/scene_graph_benchmark/output/VidORval_traj_features_th-15-5",
        traj_embds_dir = "/home/gkf/project/ALPRO/extract_features_output/VidORval_TrajFeatures256",
        traj_cls_split_info_path = "configs/VidOR_OjbectClass_spilt_info_v2.json",
        pred_cls_split_info_path = "configs/VidOR_PredClass_spilt_info_v2.json",
        use_gt = False,
    ):

        self.dataset_split = "val"
        self.use_gt = use_gt
        
        traj_cls_split_info = load_json(traj_cls_split_info_path)
        self.traj_cls2id_map = traj_cls_split_info["cls2id"]
        self.traj_cls2split_map = traj_cls_split_info["cls2split"]
        pred_cls_split_info = load_json(pred_cls_split_info_path)
        self.pred_cls2id_map = pred_cls_split_info["cls2id"]
        self.pred_cls2split_map = pred_cls_split_info["cls2split"]
        self.pred_num_base = sum([v=="base" for v in pred_cls_split_info["cls2split"].values()])
        self.pred_num_novel = sum([v=="novel" for v in pred_cls_split_info["cls2split"].values()])
        assert self.pred_num_base == 30 and self.pred_num_novel == 20


        self.dataset_dir = dataset_dir 
        self.cache_dir = cache_dir
        self.tracking_res_dir = tracking_res_dir
        # .../tracklets_results/VidORtrain_tracking_results_th-15-5/1010_8872539414/1010_8872539414-0825-0855.json
        self.traj_features_dir = traj_features_dir
        self.traj_embds_dir = traj_embds_dir
        
        
        segment_tags_all = self.prepare_segment_tags()  #
        self.segment_tags_all = segment_tags_all
        self.val_vid2videoname_map = {video_name.split('_')[1]:video_name for video_name in self.video_names}
    

    def prepare_segment_tags(self):
        self.anno_dir = os.path.join(self.dataset_dir,"annotation",_reset_dataset_split(self.dataset_split)) # .../vidor-dataset/training
        group_ids = os.listdir(self.anno_dir)
        video_names_all = []
        for gid in group_ids:
            filenames = os.listdir(os.path.join(self.anno_dir,gid))
            video_names_all  += [gid + "_" + filename.split(".")[0]  for filename in filenames]

        video_names_all = sorted(video_names_all)
        self.video_names_all = video_names_all

        
        if self.use_gt:
            filename = "VidORGTval_det_video2seg.json"            
        else:
            filename = "VidORval_det_video2seg.json"

        video2seg,video2trajs = self._get_video2seg_cache(
            filename,
            self.tracking_res_dir
        )

        seg2relidx = dict()
        segment_tags_all = []
        for video_name, seg_tags in video2seg.items():
            segment_tags_all += seg_tags
            for rel_idx,seg_tag in enumerate(seg_tags):
                seg2relidx[seg_tag] = rel_idx

        self.seg2relidx = seg2relidx
        segment_tags_all = sorted(segment_tags_all)
        
        self.video2seg = video2seg  # this is for val-set, for train-set, video2seg & video_names will be reset
        self.video_names = video_names_all


        return segment_tags_all



    def __getitem__(self, idx):
        
        video_name = self.video_names[idx]  # return video_name for debug
        segment_tags  = self.video2seg[video_name]  # some segs_might not have label
        
        traj_infos,n_trajs,n_trajs_b4sp = VidORTrajDataset.get_traj_infos(self,video_name,segment_tags,rt_ntrajs_only=False)
        traj_features,traj_embds = VidORTrajDataset.get_traj_features(self,video_name,segment_tags,n_trajs_b4sp)
        

        vlv_relpos_feats = []
        for idx,seg_tag in enumerate(segment_tags):
            traj_info = traj_infos[idx]
            # n_traj = n_trajs[idx]
            n_traj = len(traj_info["bboxes"])
            # print(seg_tag,traj_features[idx].shape[0],n_traj,traj_embds[idx].shape[0],"--------------",det2gt_ids_debug[idx])
            assert traj_features[idx].shape[0] == n_traj and n_traj == traj_embds[idx].shape[0]
            
            
            pair_ids = trajid2pairid(n_traj)  # (n_pair,2)  n_pair == n_det*(n_det-1)
            sids = pair_ids[:,0]  # (n_pair,)
            oids = pair_ids[:,1]

            if pair_ids.numel() == 0:  # if n_traj==1, no pair
                relpos_feats = torch.empty(size=(0,12))
                # LOGGER.info("empty pair_ids,n_traj={},seg_rel_idx={},seg_tag={}".format(n_traj,idx,seg_tag))
            else:
                relpos_feats = get_relative_position_feature(traj_info["fstarts"],traj_info["bboxes"],sids,oids)  # (n_pair,12)
            vlv_relpos_feats.append(relpos_feats)

        
        rt_values  = (
            segment_tags,
            traj_infos,
            traj_features,
            traj_embds,
            vlv_relpos_feats
        )

        return rt_values
    


#### datasets that modified sample manner for debias

class VidORGTDatasetForTrain_BIsample(VidORGTDatasetForTrain):
    '''
    perform the `Bi-level Data Resampling` in paper 
    Bipartite Graph Network with Adaptive Message Passing for Unbiased Scene Graph Generation (CVPR2021)
    https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Bipartite_Graph_Network_With_Adaptive_Message_Passing_for_Unbiased_Scene_CVPR_2021_paper.pdf


    '''
    def __init__(self, 
        repeat_scale = 0.07,
        droupout_scale = 0.7,
        **kargs
    ):
        super().__init__(**kargs)

        ## TODO make these two configurable
        # self.repeat_scale = 0.07  # the T in paper 
        # self.droupout_scale = 0.7  # tha gamma_d in paper

        self.repeat_scale = repeat_scale  # the T in paper 
        self.droupout_scale = droupout_scale  # tha gamma_d in paper

        self.prepare_repeat_factor()  

        self.repeat_video_names()

        LOGGER.info("------------- after BI-level sample -----------, len(self) == {}".format(len(self)))

    def prepare_repeat_factor(self):

        # pred_list = load_json("visualization_vidor/VidOR-train-base_pred_list.json")
        pred_list = []
        for seg_tag,gt_triplet in tqdm(self.gt_triplets.items(),desc="stat pred freq"):
            for spo_list in gt_triplet.values():
                preds = [spo[1] for spo in spo_list]
                pred_list += preds

        pred2counts = defaultdict(int)
        for pred in pred_list:
            pred2counts[pred]+=1
        
        assert len(pred2counts) == self.num_base

        pred2counts = sorted(pred2counts.items(),key= lambda x:x[1],reverse=True)
        pred_ins_sum = sum([x[1] for x in pred2counts])
        pred2freq = {x[0]:x[1]/pred_ins_sum for x in pred2counts}
        func_freq2repeat = lambda f:max(1,int(math.sqrt(self.repeat_scale/f)))

        repeat_factor_dict = {p:func_freq2repeat(f) for p,f in pred2freq.items()}

        LOGGER.info("pred2counts={}".format(pred2counts))
        LOGGER.info("repeat_factor_dict = {}".format(repeat_factor_dict))

        tmp = [(self.pred_cls2id_map[p],rf) for p,rf in repeat_factor_dict.items()]  
        tmp = sorted(tmp,key=lambda x:x[0])
        repeat_factor = torch.as_tensor([x[1] for x in tmp])  # (num_base,)
        repeat_factor = torch.constant_pad_nd(repeat_factor,pad=(1,self.num_novel)) # (1+num_base+num_novel,)

        self.repeat_factor = repeat_factor
        self.repeat_factor_dict = repeat_factor_dict
    
    def repeat_video_names(self):
        # 1. calculate the repeat_factor for each video according to freq of each predicate
        # 2. modify self.video_names according to repeat_factor
        videoname2rf = {video_name:0 for video_name in self.video_names}
        for seg_tag,gt_triplet in tqdm(self.gt_triplets.items(),desc="calculate the rf for each video"):
            video_name = seg_tag.split('-')[0]
            for spo_list in gt_triplet.values():
                rf = max([self.repeat_factor_dict[spo[1]] for spo in spo_list])
                if rf > videoname2rf[video_name]:
                    videoname2rf[video_name] = rf
        

        repeated_video_names = []
        for video_name in self.video_names:
            rf = videoname2rf[video_name]
            repeated_video_names += [video_name] * rf

        self.videoname2rf = videoname2rf  # # rf means repeat_factor
        self.video_names = repeated_video_names


    def __getitem__(self, idx):
        
        video_name = deepcopy(self.video_names[idx])  # return video_name for debug
        rf_crt_video = self.videoname2rf[video_name]  # repeat_factor for current video
        segment_tags  = self.video2seg[video_name]

        num_sample_segs = min(self.num_sample_segs,len(segment_tags))
        segment_tags = np.random.choice(segment_tags,num_sample_segs,replace=False)

        traj_infos,n_trajs,n_trajs_b4sp = self.get_traj_infos(video_name,segment_tags)
        traj_features,traj_embds = VidORTrajDataset.get_traj_features(self,video_name,segment_tags,n_trajs_b4sp)
        
        assert len(n_trajs) == len(traj_features) and len(n_trajs) == len(traj_embds)
        for nt,tf,te in zip(n_trajs,traj_features,traj_embds):
            assert tf.shape[0] == te.shape[0] and nt == te.shape[0]

        
        vlv_s_feats = []        # `vlv` means video-level
        vlv_o_feats = []
        vlv_s_embds = []
        vlv_o_embds = []
        vlv_relpos_feats = []
        vlv_triplet_clsids = []  
        vlv_gt_pred_vecs = []
        for i,seg_tag in enumerate(segment_tags):
            traj_info = traj_infos[i]
            # pair_ids,triplet_cls_ids = self.get_triplet_labels(self.gt_triplets[seg_tag],traj_info["tids"])
            pair_ids,gt_pred_vecs = self.get_pred_labels(self.gt_triplets[seg_tag],traj_info["tids"])
            if pair_ids.numel() == 0:
                continue
            
            ### apply instance drop_out to gt_pred_vecs (n_pair, n_cls), n_cls == 1+num_base+num_novel
            n_pair = gt_pred_vecs.shape[0]
            max_rf = torch.max(self.repeat_factor[None,:] * gt_pred_vecs,dim=-1)[0]  # (n_pair,)
            dropout_rate = (rf_crt_video - max_rf)/rf_crt_video * self.droupout_scale  # (n_pair,)
            # print(dropout_rate,"-----------------------")
            left_mask = torch.rand(dropout_rate.size()) > dropout_rate
            pair_ids = pair_ids[left_mask,:]
            gt_pred_vecs = gt_pred_vecs[left_mask,:]

            if pair_ids.numel() == 0:
                continue

            sids = pair_ids[:,0]  # (n_pair,)
            oids = pair_ids[:,1]

            
            s_feats = traj_features[i][sids,:]     # (n_pair,2048)
            o_feats = traj_features[i][oids,:]     # as above
            s_embds = traj_embds[i][sids,:]  # (n_pair,256)
            o_embds = traj_embds[i][oids,:]
            relpos_feats = get_relative_position_feature(traj_info["fstarts"],traj_info["bboxes"],sids,oids)  # (n_pair,12)

            
            vlv_s_feats.append(s_feats)
            vlv_o_feats.append(o_feats)
            vlv_s_embds.append(s_embds)
            vlv_o_embds.append(o_embds)
            vlv_relpos_feats.append(relpos_feats)
            # vlv_triplet_clsids += triplet_cls_ids
            vlv_gt_pred_vecs.append(gt_pred_vecs)
        
        if len(vlv_s_feats) == 0:
            rand_idx = np.random.choice(list(range(len(self))))
            LOGGER.info("video: {} with num_sample_segs={}, has no required data".format(video_name,num_sample_segs))
            return self.__getitem__(rand_idx)
        (
            vlv_s_feats,vlv_o_feats,vlv_s_embds,vlv_o_embds,vlv_relpos_feats,vlv_gt_pred_vecs
        ) = map(torch.cat,[vlv_s_feats,vlv_o_feats,vlv_s_embds,vlv_o_embds,vlv_relpos_feats,vlv_gt_pred_vecs]) 
        # default cat at dim=0, each shape == (N_pair, *)
        # return_items = (video_name,) + return_items + (vlv_triplet_clsids,)
        
        return_items = (
            video_name,
            vlv_s_feats,  # (N_pair, 2048), N_pair is total pairs in all segments of this video
            vlv_o_feats,
            vlv_s_embds,
            vlv_o_embds,
            vlv_relpos_feats,
            # vlv_triplet_clsids
            vlv_gt_pred_vecs
        )
        
        # video_name,s_roi_feats,o_roi_feats,s_embds,o_embds,relpos_feats,triplet_cls_ids
        return return_items

class VidORUnifiedDataset_BIsampleForTrain(VidORUnifiedDataset):
    def __init__(self, **kargs):
        super().__init__(**kargs)

        assert self.dataset_split == "train"
    
        ## TODO make these two configurable
        self.repeat_scale = 0.07  # the T in paper 
        self.droupout_scale = 0.7  # tha gamma_d in paper

        self.prepare_repeat_factor()
        self.repeat_video_names()

        LOGGER.info("------------- after BI-level sample -----------, len(self) == {}".format(len(self)))
    
    def prepare_repeat_factor(self):
        pred_list = load_json("visualization_vidor/VidOR-train-base_pred_list.json")

        pred2counts = defaultdict(int)
        for pred in pred_list:
            pred2counts[pred]+=1
        
        assert len(pred2counts) == self.pred_num_base

        pred2counts = sorted(pred2counts.items(),key= lambda x:x[1],reverse=True)
        pred_ins_sum = sum([x[1] for x in pred2counts])
        pred2freq = {x[0]:x[1]/pred_ins_sum for x in pred2counts}
        func_freq2repeat = lambda f:max(1,int(math.sqrt(self.repeat_scale/f)))

        repeat_factor_dict = {p:func_freq2repeat(f) for p,f in pred2freq.items()}

        LOGGER.info("pred2counts={}".format(pred2counts))
        LOGGER.info("repeat_factor_dict = {}".format(repeat_factor_dict))

        tmp = [(self.pred_cls2id_map[p],rf) for p,rf in repeat_factor_dict.items()]  
        tmp = sorted(tmp,key=lambda x:x[0])
        repeat_factor = torch.as_tensor([x[1] for x in tmp])  # (num_base,)
        repeat_factor = torch.constant_pad_nd(repeat_factor,pad=(1,self.pred_num_novel)) # (1+num_base+num_novel,)

        self.repeat_factor = repeat_factor
        self.repeat_factor_dict = repeat_factor_dict

    def repeat_video_names(self):
        # 1. calculate the repeat_factor for each video according to freq of each predicate
        # 2. modify self.video_names according to repeat_factor
        cache_path = os.path.join(self.cache_dir,"VidORtrain_videoname2rf.json")
        if os.path.exists(cache_path):
            videoname2rf =  load_json(cache_path)
        else:
            videoname2rf = dict()
            for video_name in tqdm(self.video_names,desc="repeat_video_names"):
                labels_dict = torch.load(os.path.join(self.label_cache_dir,video_name+".pth"))
                segment_tags = self.video2seg[video_name]
                labels = [labels_dict[seg_tag][0] for seg_tag in segment_tags]  
                labels = torch.cat(labels,dim=0) # (N_pair,n_cls) n_cls == 1 + n_base + n_novel, N_pair is total subj-obj pairs in all segs in this video
                N_pair = labels.shape[0]
                rf = self.repeat_factor[None,:].repeat(N_pair,1)
                rf = max(rf[labels.type(torch.bool)].tolist())
                videoname2rf[video_name] = rf
            with open(cache_path,'w') as f:
                json.dump(videoname2rf,f)
            LOGGER.info("videoname2rf saved at {}".format(cache_path))

        repeated_video_names = []
        for video_name in self.video_names:
            rf = videoname2rf[video_name]
            repeated_video_names += [video_name] * rf

        self.videoname2rf = videoname2rf  # # rf means repeat_factor
        self.video_names = repeated_video_names

    def negative_sampling(self,pos_mask,num_pos_left):

        # refer to test_API/test_negative_sampling.py

        neg_mask = ~pos_mask
        num_pos = num_pos_left
        num_neg = neg_mask.sum()
        
        # n_drop = num_neg - num_pos  # this can be < 0  <-- this is manually control 1:1 sample
        n_drop = (num_neg + num_pos) - int(num_pos/self.pos_ratio)

        if n_drop <= 0: 
            # when there are more pos-sample than neg-sample (n < p*(1-r)/r)
            return neg_mask
        
        idx = neg_mask.nonzero(as_tuple=True)[0]  # idx.shape == (n_neg,)
        perm = torch.randperm(num_neg)
        idx = idx[perm[:n_drop]] 
        # NOTE if n_drop < 0 , we must not use `idx[perm[:n_drop]]`, refer to refer to test_API/test_negative_sampling.py
        neg_mask[idx] = False

        return neg_mask

    def __getitem__(self, idx):
        
        video_name = self.video_names[idx]  # return video_name for debug
        rf_crt_video = self.videoname2rf[video_name]
        segment_tags  = self.video2seg[video_name]  # some segs_might not have label
        num_sample_segs = min(self.num_sample_segs,len(segment_tags))
        segment_tags = np.random.choice(segment_tags,num_sample_segs,replace=False).tolist()

        #### NOTE in get_traj_infos & get_traj_features, seg2relidx is w.r.t detection segs, 
        # (即使这里的segment_tags是从 labeled video2seg 中取出来的， 我们在get_traj_infos的时候还是去算seg2relidx is w.r.t detection segs，
        # 而不是 w.r.t labeled video2seg)
        #### And in assigned labels we must not use `self.seg2relidx`
        traj_infos,n_trajs,n_trajs_b4sp = VidORTrajDataset.get_traj_infos(self,video_name,segment_tags,rt_ntrajs_only=False)
        traj_features,traj_embds = VidORTrajDataset.get_traj_features(self,video_name,segment_tags,n_trajs_b4sp)
        
        
        if self.gt_training_traj_supp is not None:
            traj_infos,det2gt_ids,n_trajs_b4sp = self.merge_gt_trajinfo(traj_infos,video_name,segment_tags)
            # print(det2gt_ids,"-="*10)
            if det2gt_ids:   
                # i.e., if det2gt_ids not empty
                # traj_features,traj_embds,det2gt_ids_debug = self.merge_gt_trajfeatures(traj_features,traj_embds,video_name,det2gt_ids,n_trajs_b4sp)
                traj_features,traj_embds = self.merge_gt_trajfeatures(traj_features,traj_embds,video_name,det2gt_ids,n_trajs_b4sp)
        
        vlv_relpos_feats = []
        vlv_s_feats = []
        vlv_o_feats = []
        vlv_s_embds = []
        vlv_o_embds = []
        for idx,seg_tag in enumerate(segment_tags):
            traj_info = traj_infos[idx]
            # n_traj = n_trajs[idx]
            n_traj = len(traj_info["bboxes"])
            # print(seg_tag,traj_features[idx].shape[0],n_traj,traj_embds[idx].shape[0],"--------------",det2gt_ids_debug[idx])
            assert traj_features[idx].shape[0] == n_traj and n_traj == traj_embds[idx].shape[0]
            
            
            pair_ids = trajid2pairid(n_traj)  # (n_pair,2)  n_pair == n_det*(n_det-1)
            sids = pair_ids[:,0]  # (n_pair,)
            oids = pair_ids[:,1]

            if pair_ids.numel() == 0:  # if n_traj==1, no pair
                relpos_feats = torch.empty(size=(0,12))
                # LOGGER.info("empty pair_ids,n_traj={},seg_rel_idx={},seg_tag={}".format(n_traj,idx,seg_tag))
            else:
                relpos_feats = get_relative_position_feature(traj_info["fstarts"],traj_info["bboxes"],sids,oids)  # (n_pair,12)
            
            vlv_relpos_feats.append(relpos_feats)
            vlv_s_feats.append(traj_features[idx][sids,:])
            vlv_o_feats.append(traj_features[idx][oids,:])
            vlv_s_embds.append(traj_embds[idx][sids,:])
            vlv_o_embds.append(traj_embds[idx][oids,:])
        
        
        vlv_labels,vlv_masks = self._get_pred_labels(video_name,segment_tags)  # each shape (n_pair,n_cls), n_cls = 1+n_base+n_novel
        vlv_labels = torch.cat(vlv_labels,dim=0) # (N_pos_pair,n_cls), N_pos_pair is total postive subj-obj pairs in all sampled segs in this video
        vlv_masks = torch.cat(vlv_masks,dim=0)  # (N_pair,) 
        if vlv_labels.numel() == 0:  # i.e., if vlv_masks.sum() == 0
            rand_idx = np.random.choice(list(range(len(self))))
            return self.__getitem__(rand_idx)
        
        (
            vlv_s_feats,vlv_o_feats,vlv_s_embds,vlv_o_embds,vlv_relpos_feats
        ) = map(torch.cat,[vlv_s_feats,vlv_o_feats,vlv_s_embds,vlv_o_embds,vlv_relpos_feats])
        assert vlv_masks.shape[0] == vlv_s_feats.shape[0]  # i.e., N_pair


        ### apply instance drop_out to vlv_labels (N_pos_pair, n_cls), n_cls == 1+num_base+num_novel
        N_pos_pair = vlv_labels.shape[0]
        max_rf = torch.max(self.repeat_factor[None,:] * vlv_labels,dim=-1)[0]  # (N_pos_pair,)
        dropout_rate = (rf_crt_video - max_rf)/rf_crt_video * self.droupout_scale  # (N_pos_pair,)
        # print(dropout_rate,"-----------------------")
        left_mask = torch.rand(dropout_rate.size()) > dropout_rate
        vlv_labels = vlv_labels[left_mask,:]
        num_pos_left = left_mask.sum()
        
        if num_pos_left == 0:  
            rand_idx = np.random.choice(list(range(len(self))))
            return self.__getitem__(rand_idx)
        
        # vlv_labels = torch.constant_pad_nd(vlv_labels,pad=(0,0,0,neg_mask.sum()))
        _func = lambda x : x[vlv_masks,:][left_mask,:]
        (
            vlv_s_feats,vlv_o_feats,vlv_s_embds,vlv_o_embds,vlv_relpos_feats
        ) = map(_func,[vlv_s_feats,vlv_o_feats,vlv_s_embds,vlv_o_embds,vlv_relpos_feats])
        assert vlv_labels.shape[0] == vlv_s_feats.shape[0]  # i.e., N_pair_after_sample

        ### sample pairs
        (
            vlv_s_feats,vlv_o_feats,vlv_s_embds,vlv_o_embds,vlv_relpos_feats,vlv_labels
        ) = map(self.pair_sampling,[vlv_s_feats,vlv_o_feats,vlv_s_embds,vlv_o_embds,vlv_relpos_feats,vlv_labels])
        # print(vlv_s_feats.shape)
        rt_values = (
            video_name,
            vlv_s_feats,
            vlv_o_feats,
            vlv_s_embds,
            vlv_o_embds,
            vlv_relpos_feats,
            vlv_labels
        )

        return rt_values

#### dataset that return relative GIoU for motion-based grouped prompt

class VidORGTDatasetForTrain_BIsample_GIoU(VidORGTDatasetForTrain_BIsample):
    
    def __getitem__(self, idx):
        
        video_name = deepcopy(self.video_names[idx])  # return video_name for debug
        rf_crt_video = self.videoname2rf[video_name]  # repeat_factor for current video
        segment_tags  = self.video2seg[video_name]

        num_sample_segs = min(self.num_sample_segs,len(segment_tags))
        segment_tags = np.random.choice(segment_tags,num_sample_segs,replace=False)

        traj_infos,n_trajs,n_trajs_b4sp = self.get_traj_infos(video_name,segment_tags)
        traj_features,traj_embds = VidORTrajDataset.get_traj_features(self,video_name,segment_tags,n_trajs_b4sp)
        
        assert len(n_trajs) == len(traj_features) and len(n_trajs) == len(traj_embds)
        for nt,tf,te in zip(n_trajs,traj_features,traj_embds):
            assert tf.shape[0] == te.shape[0] and nt == te.shape[0]

        
        vlv_s_feats = []        # `vlv` means video-level
        vlv_o_feats = []
        vlv_s_embds = []
        vlv_o_embds = []
        vlv_relpos_feats = []
        vlv_rel_gious = []
        vlv_triplet_clsids = []  
        vlv_gt_pred_vecs = []
        for i,seg_tag in enumerate(segment_tags):
            traj_info = traj_infos[i]
            # pair_ids,triplet_cls_ids = self.get_triplet_labels(self.gt_triplets[seg_tag],traj_info["tids"])
            pair_ids,gt_pred_vecs = self.get_pred_labels(self.gt_triplets[seg_tag],traj_info["tids"])
            if pair_ids.numel() == 0:
                continue
            
            ### apply instance drop_out to gt_pred_vecs (n_pair, n_cls), n_cls == 1+num_base+num_novel
            n_pair = gt_pred_vecs.shape[0]
            max_rf = torch.max(self.repeat_factor[None,:] * gt_pred_vecs,dim=-1)[0]  # (n_pair,)
            dropout_rate = (rf_crt_video - max_rf)/rf_crt_video * self.droupout_scale  # (n_pair,)
            # print(dropout_rate,"-----------------------")
            left_mask = torch.rand(dropout_rate.size()) > dropout_rate
            pair_ids = pair_ids[left_mask,:]
            gt_pred_vecs = gt_pred_vecs[left_mask,:]

            if pair_ids.numel() == 0:
                continue

            sids = pair_ids[:,0]  # (n_pair,)
            oids = pair_ids[:,1]

            
            s_feats = traj_features[i][sids,:]     # (n_pair,2048)
            o_feats = traj_features[i][oids,:]     # as above
            s_embds = traj_embds[i][sids,:]  # (n_pair,256)
            o_embds = traj_embds[i][oids,:]
            relpos_feats = get_relative_position_feature(traj_info["fstarts"],traj_info["bboxes"],sids,oids)  # (n_pair,12)
            rel_gious = _get_traj_pair_GIoU(traj_info["bboxes"],sids,oids)  # (n_pair,2)
            
            vlv_s_feats.append(s_feats)
            vlv_o_feats.append(o_feats)
            vlv_s_embds.append(s_embds)
            vlv_o_embds.append(o_embds)
            vlv_relpos_feats.append(relpos_feats)
            vlv_rel_gious.append(rel_gious)
            # vlv_triplet_clsids += triplet_cls_ids
            vlv_gt_pred_vecs.append(gt_pred_vecs)
        
        if len(vlv_s_feats) == 0:
            rand_idx = np.random.choice(list(range(len(self))))
            LOGGER.info("video: {} with num_sample_segs={}, has no required data".format(video_name,num_sample_segs))
            return self.__getitem__(rand_idx)
        (
            vlv_s_feats,vlv_o_feats,vlv_s_embds,vlv_o_embds,vlv_relpos_feats,vlv_rel_gious,vlv_gt_pred_vecs
        ) = map(torch.cat,[vlv_s_feats,vlv_o_feats,vlv_s_embds,vlv_o_embds,vlv_relpos_feats,vlv_rel_gious,vlv_gt_pred_vecs]) 
        # default cat at dim=0, each shape == (N_pair, *)
        # return_items = (video_name,) + return_items + (vlv_triplet_clsids,)
        
        return_items = (
            video_name,
            vlv_s_feats,  # (N_pair, 2048), N_pair is total pairs in all segments of this video
            vlv_o_feats,
            vlv_s_embds,
            vlv_o_embds,
            vlv_relpos_feats,
            vlv_rel_gious,
            # vlv_triplet_clsids
            vlv_gt_pred_vecs
        )
        
        # video_name,s_roi_feats,o_roi_feats,s_embds,o_embds,relpos_feats,triplet_cls_ids
        return return_items


class VidORUnifiedDataset_ForEval_GIoU(VidORUnifiedDataset_ForEval):
    
    def __getitem__(self, idx):
        
        video_name = self.video_names[idx]  # return video_name for debug
        segment_tags  = self.video2seg[video_name]  # some segs_might not have label
        
        traj_infos,n_trajs,n_trajs_b4sp = VidORTrajDataset.get_traj_infos(self,video_name,segment_tags,rt_ntrajs_only=False)
        traj_features,traj_embds = VidORTrajDataset.get_traj_features(self,video_name,segment_tags,n_trajs_b4sp)
        

        vlv_relpos_feats = []  # vlv means video-level
        vlv_rel_gious = []
        for idx,seg_tag in enumerate(segment_tags):
            traj_info = traj_infos[idx]
            # n_traj = n_trajs[idx]
            n_traj = len(traj_info["bboxes"])
            # print(seg_tag,traj_features[idx].shape[0],n_traj,traj_embds[idx].shape[0],"--------------",det2gt_ids_debug[idx])
            assert traj_features[idx].shape[0] == n_traj and n_traj == traj_embds[idx].shape[0]
            
            
            pair_ids = trajid2pairid(n_traj)  # (n_pair,2)  n_pair == n_det*(n_det-1)
            sids = pair_ids[:,0]  # (n_pair,)
            oids = pair_ids[:,1]

            if pair_ids.numel() == 0:  # if n_traj==1, no pair
                relpos_feats = torch.empty(size=(0,12))
                rel_gious = torch.empty(size=(0,2))
                # LOGGER.info("empty pair_ids,n_traj={},seg_rel_idx={},seg_tag={}".format(n_traj,idx,seg_tag))
            else:
                relpos_feats = get_relative_position_feature(traj_info["fstarts"],traj_info["bboxes"],sids,oids)  # (n_pair,12)
                rel_gious = _get_traj_pair_GIoU(traj_info["bboxes"],sids,oids)
            vlv_relpos_feats.append(relpos_feats)
            vlv_rel_gious.append(rel_gious)

        
        rt_values  = (
            segment_tags,
            traj_infos,
            traj_features,
            traj_embds,
            vlv_relpos_feats,
            vlv_rel_gious
        )

        return rt_values
    

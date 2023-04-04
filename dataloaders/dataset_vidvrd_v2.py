 


import os
import json
import pickle
from collections import defaultdict

from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

from utils.logger import LOGGER
from utils.utils_func import vIoU_broadcast,vPoI_broadcast,trajid2pairid,bbox_GIoU


def load_json(filename):
    with open(filename, "r") as f:
        x = json.load(f)
    return x


def prepare_segment_tags(dataset_dir,tracking_res_dir,dataset_splits):
    '''
    TODO wirte more elegant code for this func
    '''
    video_name_to_split = dict()
    for split in ["train","test"]:
        anno_dir = os.path.join(dataset_dir,split)
        for filename in sorted(os.listdir(anno_dir)):
            video_name = filename.split('.')[0]  # e.g., ILSVRC2015_train_00405001.json
            video_name_to_split[video_name] = split
    
    segment_tags_all = [x.split('.')[0] for x in sorted(os.listdir(tracking_res_dir))]  # e.g., ILSVRC2015_train_00010001-0016-0048.json
    segment_tags = []
    for seg_tag in segment_tags_all:
        video_name = seg_tag.split('-')[0] # e.g., ILSVRC2015_train_00010001-0016-0048
        split = video_name_to_split[video_name]
        if split in dataset_splits:
            segment_tags.append(seg_tag)
    
    return segment_tags


def _to_xywh(bboxes):
    x = (bboxes[...,0] + bboxes[...,2])/2
    y = (bboxes[...,1] + bboxes[...,3])/2
    w = bboxes[...,2] - bboxes[...,0]
    h = bboxes[...,3] - bboxes[...,1]
    return x,y,w,h



class VidVRDTrajDataset(object):
    '''
    Here we filter out traj (based on traj_len_th & min_region_th ) directly after Seq-NMS and store as json
    '''
    def __init__(self,
        class_splits,
        dataset_split,
        class_spilt_info_path = "/home/gkf/project/VidVRD-OpenVoc/configs/VidVRD_class_spilt_info.json",
        dataset_dir = "/home/gkf/project/VidVRD_VidOR/vidvrd-dataset",
        tracking_res_dir = "/home/gkf/project/VidVRD-II/tracklets_results/VidVRD_segment30_tracking_results_th-15-5",
        traj_features_dir = "/home/gkf/project/scene_graph_benchmark/output/VidVRD_traj_features_seg30_th-15-5",
        traj_embeddings_dir = "/home/gkf/project/ALPRO/extract_features_output/vidvrd_seg30_TrajFeatures256",
        cache_dir = "datasets/cache_vidvrd",
        vIoU_th = 0.5
    ):
        super().__init__()

        self.vIoU_th = vIoU_th
        self.class_splits = tuple(cs.lower() for cs in class_splits)   # e.g., ("base","novel"), or ("base",) or ("novel",)
        self.dataset_split = dataset_split.lower()# e.g., "train" or "test"
        assert self.dataset_split in ("train","test")
        with open(class_spilt_info_path,'r') as f:
            self.class_split_info = json.load(f)

        self.dataset_dir = dataset_dir  #  train & test, # e.g., ILSVRC2015_train_00405001.json
        self.tracking_res_dir = tracking_res_dir                
        self.traj_features_dir = traj_features_dir        
        self.traj_embeddings_dir = traj_embeddings_dir    
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        

        self.segment_tags = self.prepare_segment_tags()
        self.annotations = self.get_anno()

        if self.dataset_split == "train":
            assert self.class_splits == ("base",)
            path = os.path.join(self.cache_dir,"VidVRDtrain_DetTrajAssignedLabels_vIoU-{:.2f}.pth".format(self.vIoU_th))  
            ### NOTE only base labels
            if os.path.exists(path):
                LOGGER.info("assigned_labels load from {}".format(path))
                assigned_labels = torch.load(path)
            else:
                assigned_labels = self.cls_label_assignment()  # (n_det,)
                torch.save(assigned_labels,path)
                LOGGER.info("assigned_labels saved at {}".format(path))

            n_total = len(assigned_labels)
            assigned_labels = {k:v for k,v in assigned_labels.items() if v is not None} 
            self.segment_tags = sorted(assigned_labels.keys())
            self.assigned_labels = assigned_labels
            
            LOGGER.info("filterout labels with None, total:{}, left:{}".format(n_total,len(self.segment_tags)))
            del self.annotations
        
        LOGGER.info(" ---------------- dataset constructed len(self) == {} ----------------".format(len(self)))
    
    def __len__(self):

        return len(self.segment_tags)


    def prepare_segment_tags(self):
        '''
        TODO wirte more elegant code for this func
        '''
        print("preparing segment_tags for data_split: {}".format(self.dataset_split),end="... ")
        video_name_to_split = dict()
        for split in ["train","test"]:
            anno_dir = os.path.join(self.dataset_dir,split)
            for filename in sorted(os.listdir(anno_dir)):
                video_name = filename.split('.')[0]  # e.g., ILSVRC2015_train_00405001.json
                video_name_to_split[video_name] = split
        
        segment_tags_all = [x.split('.')[0] for x in sorted(os.listdir(self.tracking_res_dir))]  # e.g., ILSVRC2015_train_00010001-0015-0045.json
        segment_tags = []
        for seg_tag in segment_tags_all:
            video_name = seg_tag.split('-')[0] # e.g., ILSVRC2015_train_00010001-0015-0045
            split = video_name_to_split[video_name]
            if split == self.dataset_split:
                segment_tags.append(seg_tag)
        print("total: {}".format(len(segment_tags)))

        return segment_tags

    def get_traj_infos(self,seg_tag):
        '''
        tracking_results = [
            {
                'fstarts': int(fstart),      # relative frame idx w.r.t this segment
                'score': score,             # float scalar
                'bboxes': bboxes.tolist()   # list[list], len == num_frames, format xyxy
                'label':  cls_id            # int, w.r.t VinVL classes
            },
            {
                'fstart': int(fstart),      # relative frame idx w.r.t this segment
                'score': score,             # float scalar
                'bboxes': bboxes.tolist()   # list[list], len == num_frames, format xyxy
                'label':  cls_id            # int, w.r.t VinVL classes
            },
            ...
        ]  # len(tracking_results) == num_tracklets
        '''
        
        path = os.path.join(self.tracking_res_dir,seg_tag + ".json")
        with open(path,'r') as f:
            tracking_results = json.load(f)

        fstarts = []
        scores = []
        bboxes = []
        cls_ids = []
        for ii,res in enumerate(tracking_results):

            
            fstarts.append(res["fstart"])  # relative frame_id  w.r.t segment fstart
            scores.append(res["score"])
            bboxes.append(torch.as_tensor(res["bboxes"])) # (num_boxes, 4)
            cls_ids.append(res["label"])
            
            

        traj_infos = {
            "fstarts":torch.as_tensor(fstarts), # shape == (n_det,)
            "scores":torch.as_tensor(scores),  # shape == (n_det,)
            "bboxes":bboxes,  # list[tensor] , len== n_det, each shape == (num_boxes, 4)
            "VinVL_clsids":torch.as_tensor(cls_ids),  # shape == (n_det,)
        }
            
        
        return traj_infos


    def get_traj_features(self,seg_tag):

        ### 2048-d RoI feature
        path = os.path.join(self.traj_features_dir,seg_tag+'.npy')
        temp = np.load(path).astype('float32')  # (num_traj, 2048)
        traj_features = torch.from_numpy(temp)

        ### 256-d Alpro-visual embedding
        path = os.path.join(self.traj_embeddings_dir,seg_tag+'.npy')
        temp = np.load(path).astype('float32')  # (num_traj, 1+256)
        temp = torch.from_numpy(temp)
        traj_ids = temp[:,0]        # (num_traj,)
        traj_embds = temp[:,1:]  # (num_traj, 256)
        
        assert traj_embds.shape[0] == traj_features.shape[0]
        return traj_features,traj_embds


    def get_anno(self):
        # self.class_split_info
        '''
            class_split_info = {
                "cls2id":{"person": 1, "dog": 2, ...},
                "id2cls":{1: "person", 2: "dog", ...},
                "cls2split":{"person":"base",...,"watercraft":"novel",...}
            }
        '''
        
        print("preparing annotations for data_split: {}, class_splits: {} ".format(self.dataset_split,self.class_splits))

        # avoid loading the same video's annotations multiple times
        annos = dict()
        anno_dir = os.path.join(self.dataset_dir,self.dataset_split)
        for filename in sorted(os.listdir(anno_dir)):
            video_name = filename.split('.')[0]  # e.g., ILSVRC2015_train_00405001.json
            path = os.path.join(anno_dir,filename)

            with open(path,'r') as f:
                anno_per_video = json.load(f)  # refer to `datasets/vidvrd-dataset/format.py`
            annos[video_name] = anno_per_video
        
        segment2anno_map = dict()
        for seg_tag in tqdm(self.segment_tags):  #NOTE 这里包括 test set， 但是annos可能只有train
            video_name, fstart, fend = seg_tag.split('-')  # e.g., "ILSVRC2015_train_00010001-0015-0045"
            fstart,fend = int(fstart),int(fend)
            anno = annos[video_name]
            trajid2cls_map = {traj["tid"]:traj["category"] for traj in anno["subject/objects"]}
            # e.g., {0: 'bicycle', 1: 'bicycle', 2: 'person', 3: 'person', 5: 'person'} # not necessarily continuous

            trajs_info = {tid:defaultdict(list) for tid in trajid2cls_map.keys()}      
            annotated_len = len(anno["trajectories"])
            
            for frame_id in range(fstart,fend,1):  # 75， 105
                if frame_id >= annotated_len:  
                    # e.g., for "ILSVRC2015_train_00008005-0075-0105", annotated_len==90, and anno_per_video["frame_count"] == 130
                    break

                frame_anno = anno["trajectories"][frame_id]  # NOTE frame_anno can be [] (empty) for all `fstart` to `fend`
                # print(seg_tag,frame_id,len(frame_anno))
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
                split_ = self.class_split_info["cls2split"][class_]
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
            
            segment2anno_map[seg_tag] = traj_annos
        
       
        return segment2anno_map
    


    
    def cls_label_assignment(self):
        print("constructing trajectory classification labels ... (vIoU_th={})".format(self.vIoU_th))
        
        assigned_labels = dict()

        for seg_tag in tqdm(self.segment_tags):
            det_info = self.get_traj_infos(seg_tag)
            det_trajs = det_info["bboxes"]    # list[tensor] , len== n_det, each shape == (num_boxes, 4)
            det_fstarts = det_info["fstarts"]  # (n_det,)

            gt_anno = self.annotations[seg_tag]
            if gt_anno is None:
                assigned_labels[seg_tag] = None
                continue


            gt_trajs = gt_anno["bboxes"]      # list[tensor] , len== n_gt,  each shape == (num_boxes, 4)
            gt_fstarts = gt_anno["fstarts"]   # (n_gt,)
            gt_labels = gt_anno["labels"]     # (n_gt,)  # range: 1~num_base
            n_gt = len(gt_labels)

            viou_matrix = vIoU_broadcast(det_trajs,gt_trajs,det_fstarts,gt_fstarts)  # (n_det, n_gt)

            max_vious, gt_ids = torch.max(viou_matrix,dim=-1)  # shape == (n_det,)
            mask = max_vious > self.vIoU_th
            gt_ids[~mask] = n_gt
            gt_labels_with_bg = torch.constant_pad_nd(gt_labels,pad=(0,1),value=0) # (n_gt+1,)
            assigned_labels[seg_tag] = gt_labels_with_bg[gt_ids]  # (n_det,)
        
        return assigned_labels

    
    
    def __getitem__(self,idx):
        seg_tag = deepcopy(self.segment_tags[idx])   # return seg_tag for debug

        traj_features, traj_embeddings = self.get_traj_features(seg_tag)
        if self.dataset_split == "train":
            traj_infos = None
            gt_annos = None
            labels = deepcopy(self.assigned_labels[seg_tag])
        else:
            traj_infos = self.get_traj_infos(seg_tag)
            gt_annos = deepcopy(self.annotations[seg_tag])
            labels = None



        return seg_tag, traj_infos, traj_features, traj_embeddings,gt_annos,labels
    
        
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
    


class VidVRDGTDatasetForTrain(object):
    def __init__(self,
        dataset_dir = "/home/gkf/project/VidVRD_VidOR/vidvrd-dataset",
        tracking_results_dir = "/home/gkf/project/scene_graph_benchmark/output/VidVRD_tracking_results_gt",
        traj_features_dir = "/home/gkf/project/scene_graph_benchmark/output/VidVRD_gt_traj_features_seg30",
        traj_embd_dir = "/home/gkf/project/ALPRO/extract_features_output/vidvrd_seg30_TrajFeatures256_gt",
        traj_cls_split_info_path = "TODO",
        pred_cls_split_info_path = "TODO",
        pred_cls_splits = ("base",),
        traj_cls_splits = ("base",),
        bsz_wrt_pair = True,
    ):

        self.dataset_dir = dataset_dir
        self.tracking_results_dir = tracking_results_dir
        self.traj_features_dir = traj_features_dir
        self.traj_embd_dir = traj_embd_dir
        traj_cls_split_info = load_json(traj_cls_split_info_path)
        self.traj_cls2id_map = traj_cls_split_info["cls2id"]
        self.traj_cls2split_map = traj_cls_split_info["cls2split"]
        pred_cls_split_info = load_json(pred_cls_split_info_path)
        self.pred_cls2id_map = pred_cls_split_info["cls2id"]
        self.pred_cls2split_map = pred_cls_split_info["cls2split"]
        self.pred_cls_splits = pred_cls_splits
        self.traj_cls_splits = traj_cls_splits
        self.bsz_wrt_pair = bsz_wrt_pair

        segment_tags = [filename.split('.')[0] for filename in  sorted(os.listdir(self.tracking_results_dir))] # e.g., ILSVRC2015_train_00010001-0016-0048
        self.segment_tags = segment_tags  # 5855 for vidvrd-train

        self.traj_infos = self.get_traj_infos()
        self.gt_triplets = self.get_gt_triplets()  # filter out segs that have no relation annotation
        self.segment_tags = sorted(self.gt_triplets.keys())  
        # 3031 for pred_cls_splits = ("base","novel") & traj_cls_splits = ("base","novel")
        # 2981 for pred_cls_splits = ("base",) & traj_cls_splits = ("base","novel")
        # 1995 for pred_cls_splits = ("base",) & traj_cls_splits = ("base",)

        if self.bsz_wrt_pair:
            print("merge data from all segments ...")
            all_datas = []
            for seg_tag in tqdm(self.segment_tags):
                data = self.getitem_for_seg(seg_tag)
                # ( 
                #     seg_tag,
                #     s_roi_feats,    # (n_pair,2048)
                #     o_roi_feats,
                #     s_embds,        # （n_pair,256)
                #     o_embds,
                #     relpos_feats,   # (n_pair,12)
                #     triplet_cls_ids # list[tesor], len == n_pair, each shape == (n_preds,3)
                # ) = data
                n_pair = data[1].shape[0]
                for i in range(n_pair):
                    single_pair_data = tuple(x[i] for x in data[1:])
                    all_datas.append(single_pair_data)
            self.all_datas = all_datas
            

        print("------------- dataset constructed -----------, len(self) == {}".format(len(self)))
    
    def __len__(self):
        if self.bsz_wrt_pair:
            return len(self.all_datas)
        else:
            return len(self.segment_tags)

    def load_video_annos(self):
        print("loading video annos for train set ...")
        # avoid loading the same video's annotations multiple times (i.e., for multiple segments in a same video)
        annos = dict()
        anno_dir = os.path.join(self.dataset_dir,"train")
        for filename in sorted(os.listdir(anno_dir)):
            video_name = filename.split('.')[0]  # e.g., ILSVRC2015_train_00405001.json
            path = os.path.join(anno_dir,filename)

            with open(path,'r') as f:
                anno_per_video = json.load(f)  # refer to `datasets/vidvrd-dataset/format.py`
            annos[video_name] = anno_per_video

        return annos
    

    def get_traj_embds(self,seg_tag):
        
        path = os.path.join(self.traj_embd_dir,seg_tag+'.npy')
        temp = np.load(path).astype('float32')  # (n_gt_traj, 1+256)
        temp = torch.from_numpy(temp)
        traj_embds = temp[:,1:]    # (n_gt_traj, 256)
        # traj_ids = temp[:,0]        # (n_gt_traj,)  we do not filter gt_traj w.r.t traj_len_th
        
        return traj_embds


    def get_gt_triplets(self):
        
        video_annos = self.load_video_annos()

        gt_triplets_all = dict()
        for seg_tag in tqdm(self.segment_tags):
            video_name, seg_fs, seg_fe = seg_tag.split('-')  # e.g., "ILSVRC2015_train_00010001-0016-0048"
            seg_fs,seg_fe = int(seg_fs),int(seg_fe)
            anno = deepcopy(video_annos[video_name])
            trajid2cls_map = {traj["tid"]:traj["category"] for traj in anno["subject/objects"]}
            # e.g., {0: 'bicycle', 1: 'bicycle', 2: 'person', 3: 'person', 5: 'person'} # not necessarily continuous

            gt_triplets = defaultdict(list)
            relations = anno["relation_instances"]
            for rel in relations: # loop for relations of all segments in this video, some segments may have no annotation
                s_tid,o_tid = rel["subject_tid"],rel["object_tid"]
                s_cls = trajid2cls_map[s_tid]
                o_cls = trajid2cls_map[o_tid]
                pred_cls = rel["predicate"]   # pred class name
                if not (self.pred_cls2split_map[pred_cls] in self.pred_cls_splits):
                    continue
                if (self.traj_cls2split_map[s_cls] not in self.traj_cls_splits) or (self.traj_cls2split_map[o_cls] not in self.traj_cls_splits):
                    continue
                
                fs,fe =  rel["begin_fid"],rel["end_fid"]  # [fs,fe)  fe is exclusive (from annotation)
                if not (seg_fs <= fs and fe <= seg_fe):  # we only select predicate that within this segment <-- actually, I suppose that this operation is WRONG
                    continue
                assert seg_fs == fs and seg_fe == fe     # 由于VidVRD标注的原因，每个relation都是标注一整个segment的 （对VidOR 不适用）
                
                gt_triplets[(s_tid,o_tid)].append((s_cls,pred_cls,o_cls))
            if len(gt_triplets) > 0:
                gt_triplets_all[seg_tag] = gt_triplets
            
        return gt_triplets_all

    def get_traj_infos(self):
        info_str = "loading tracking_results from {} ... ".format(self.tracking_results_dir)
        # refer to func:reformulate_gt_to_tracking_results_format() in `/home/gkf/project/scene_graph_benchmark/tools/extract_features/extract_traj_features.py`
        print(info_str)
        '''
        tracking_results = [
            {
                'fstarts': int(fstart),      # relative frame idx w.r.t this segment
                'score': score,             # float scalar
                'bboxes': bboxes.tolist()   # list[list], len == num_frames, format xyxy
                'class':  class_name         
                'tid':  tid
            },
            ...
        ]  # len(tracking_results) == num_tracklets
        '''
        traj_infos = dict()
        for seg_tag in tqdm(self.segment_tags):

            path = os.path.join(self.tracking_results_dir,seg_tag + ".json")
            with open(path,'r') as f:
                gt_results = json.load(f)
            path = os.path.join(self.traj_features_dir,seg_tag+".npy")
            gt_features = np.load(path).astype('float32')  # (num_traj, 2048)
            
            fstarts = []
            scores = []
            bboxes = []
            tids = []
            clsids = []
            for res in gt_results:
                # print(res.keys())
                fstarts.append(res["fstart"])  # relative frame_id  w.r.t segment fstart
                scores.append(res["score"])
                bboxes.append(torch.as_tensor(res["bboxes"])) # (num_boxes, 4)
                tids.append(res["tid"])
                clsids.append(self.traj_cls2id_map[res["class"]])
            
            if scores:  # i.e., if scores != []
                traj_info = {
                    "fstarts":torch.as_tensor(fstarts), # shape == (num_traj,)
                    "scores":torch.as_tensor(scores),  # shape == (num_traj,)
                    "bboxes":bboxes,  # list[tensor] , len== num_traj, each shape == (num_boxes, 4)
                    "features":torch.from_numpy(gt_features),  # shape == (num_traj, 2048)
                    "tids":torch.as_tensor(tids),        # shape == (num_traj,)
                    "clsids":torch.as_tensor(clsids)    # object category id
                }
            else:
                traj_info = None
            traj_infos[seg_tag] = traj_info
        
        return traj_infos


    def _to_xywh(self,bboxes):
        x = (bboxes[...,0] + bboxes[...,2])/2
        y = (bboxes[...,1] + bboxes[...,3])/2
        w = bboxes[...,2] - bboxes[...,0]
        h = bboxes[...,3] - bboxes[...,1]
        return x,y,w,h

    def get_pred_labels(self,gt_triplets,traj_tids):
        '''TODO FIXME add background label for negative pairs'''
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
        
        # print(list(gt_triplets.keys()),pair_ids.shape)
        # triplet_cls_ids len == n_pair each shape == (n_preds,3)

        return pair_ids,triplet_cls_ids
    
    def __getitem__(self, idx):
        
        if not self.bsz_wrt_pair:
            return self.getitem_for_seg(self.segment_tags[idx])
        

        single_pair_data = deepcopy(self.all_datas[idx])
        # (
        #     s_roi_feats,  # (2048,)
        #     o_roi_feats,
        #     s_embds,      # (256,)
        #     o_embds
        #     relpos_feats, # (12,)
        #     triplet_cls_ids   # (n_preds,3)
        # ) = single_pair_data
        # TODO FIXME add negative sample
        return single_pair_data
        

    def getitem_for_seg(self,seg_tag):
        seg_tag = deepcopy(seg_tag)
        gt_triplets = deepcopy(self.gt_triplets[seg_tag])
        traj_infos = deepcopy(self.traj_infos[seg_tag])
        pair_ids,triplet_cls_ids = self.get_pred_labels(gt_triplets,traj_infos["tids"])

        sids = pair_ids[:,0]
        oids = pair_ids[:,1]

        relpos_feats = self.get_relative_position_feature(seg_tag,sids,oids)  # (n_pair,12)

        traj_features  = traj_infos["features"]
        s_roi_feats = traj_features[sids,:]     # (n_pair,2048)
        o_roi_feats = traj_features[oids,:]     # as above

        traj_embds = self.get_traj_embds(seg_tag)
        s_embds = traj_embds[sids,:]  # (n_pair,256)
        o_embds = traj_embds[oids,:]


        return seg_tag,s_roi_feats,o_roi_feats,s_embds,o_embds,relpos_feats,triplet_cls_ids
 

    def get_relative_position_feature(self,seg_tag,sids,oids):

        traj_fstarts = self.traj_infos[seg_tag]["fstarts"]  # (n_det,)
        traj_bboxes = self.traj_infos[seg_tag]["bboxes"] # list[tensor] , len == n_det, each shape == (num_boxes, 4)
        

        ## 2.
        s_trajs = [traj_bboxes[idx] for idx in sids]  # format: xyxy
        o_trajs = [traj_bboxes[idx] for idx in oids]  # len == n_pair, each shape == (n_frames,4)

        s_fstarts = traj_fstarts[sids]  # (n_pair_aft_filter,)
        o_fstarts = traj_fstarts[oids]  # 

        s_lens = torch.as_tensor([x.shape[0] for x in s_trajs],device=s_fstarts.device)  # (n_pair_aft_filter,)
        o_lens = torch.as_tensor([x.shape[0] for x in o_trajs],device=o_fstarts.device)

        s_duras = torch.stack([s_fstarts,s_fstarts+s_lens],dim=-1)  # (n_pair_aft_filter,2)
        o_duras = torch.stack([o_fstarts,o_fstarts+o_lens],dim=-1)  # (n_pair_aft_filter,2)

        s_bboxes = [torch.stack((boxes[0,:],boxes[-1,:]),dim=0) for boxes in s_trajs]  # len == n_pair_aft_filter, each shape == (2,4)
        s_bboxes = torch.stack(s_bboxes,dim=0)  # (n_pair_aft_filter, 2, 4)  # 2 stands for the start & end bbox of the traj

        o_bboxes = [torch.stack((boxes[0,:],boxes[-1,:]),dim=0) for boxes in o_trajs]
        o_bboxes = torch.stack(o_bboxes,dim=0)  # (n_pair_aft_filter, 2, 4)

    
        ## 3. calculate relative position feature
        subj_x, subj_y, subj_w, subj_h = self._to_xywh(s_bboxes.float())  # (n_pair_aft_filter,2)
        obj_x, obj_y, obj_w, obj_h = self._to_xywh(o_bboxes.float())      # (n_pair_aft_filter,2)

        log_subj_w, log_subj_h = torch.log(subj_w), torch.log(subj_h)
        log_obj_w, log_obj_h = torch.log(obj_w), torch.log(obj_h)

        rx = (subj_x-obj_x)/obj_w   # (n_pair_aft_filter,2)
        ry = (subj_y-obj_y)/obj_h
        rw = log_subj_w-log_obj_w
        rh = log_subj_h-log_obj_h
        ra = log_subj_w+log_subj_h-log_obj_w-log_obj_h
        rt = (s_duras-o_duras) / 30  # (n_pair_aft_filter,2)
        rel_pos_feat = torch.cat([rx,ry,rw,rh,ra,rt],dim=-1)  # (n_pair_aft_filter,12)

        return rel_pos_feat
        
    def get_collator_func(self):
        
        def collate_func1(batch_data):
            bsz = len(batch_data)
            n_items = len(batch_data[0])
            # assert n_items == 6   # exclude seg_tag
            batch_data = [[batch_data[bid][iid] for bid in range(bsz)] for iid in range(n_items)]
            batch_data = tuple(torch.stack(bath_di,dim=0) if i<n_items-1 else bath_di for i,bath_di in enumerate(batch_data))
            # the last one is triplet_cls_ids
            return batch_data
        
        def collate_func2(batch_data):
            # this collator_func simply swaps the order of inner and outer of batch_data

            bsz = len(batch_data)
            num_rt_vals = len(batch_data[0])
            # assert num_rt_vals == 7
            return_values = tuple([batch_data[bid][vid] for bid in range(bsz)] for vid in range(num_rt_vals))
            return return_values


        if self.bsz_wrt_pair:
            return collate_func1
        else:
            return collate_func2


class VidVRDUnifiedDataset(object):
    '''
    NOTE
    unlike `VidVRDTrajDataset`, this dataset does not construct gt_anno, 
    because we directly use VidVRD-helper (VidVRD-II) (which include dataset & gt_anno construction) to evaluate the raltion detection mAP & recall
    Therefore, we also do not set cls_split for relation ("base", "novel" or "all")
    (for training, we select the cls_slpit for label in `RelationClsModel`, for test, we set cls_split in evaluate func in VidVRD-II)
    '''
    def __init__(self,
        dataset_splits,
        enti_cls_spilt_info_path = "/home/gkf/project/VidVRD-OpenVoc/configs/VidVRD_class_spilt_info.json",
        pred_cls_split_info_path = "xxx",  # not used for test
        dataset_dir = "/home/gkf/project/VidVRD_VidOR/vidvrd-dataset",
        traj_info_dir = "/home/gkf/project/VidVRD-II/tracklets_results/VidVRD_segment30_tracking_results",  # all 1000 videos
        traj_features_dir = "/home/gkf/project/scene_graph_benchmark/output/VidVRD_traj_features_seg30", # 2048-d RoI feature
        traj_embd_dir = "/home/gkf/project/ALPRO/extract_features_output/vidvrd_seg30_TrajFeatures256", ## 256-d
        cache_dir = "/home/gkf/project/VidVRD-OpenVoc/datasets/cache",
        gt_training_traj_supp = dict(
            traj_dir = "/home/gkf/project/scene_graph_benchmark/output/VidVRD_tracking_results_gt",
            feature_dir = "/home/gkf/project/scene_graph_benchmark/output/VidVRD_gt_traj_features_seg30",
            embd_dir = "/home/gkf/project/ALPRO/extract_features_output/vidvrd_seg30_TrajFeatures256_gt",
        ),
        pred_cls_splits = ("base",), # only used for train
        traj_cls_splits = ("base",), # only used for train
        traj_len_th = 15,
        min_region_th = 5,
        vpoi_th = 0.9,
        cache_tag = "",
        assign_label = None, # None means not specified
        ):
        self.dataset_splits = tuple(ds.lower() for ds in dataset_splits)  # e.g., ("train","test"), or ("train",) or ("test",)
        self.traj_cls_splits = traj_cls_splits # we do not assign traj labels here, but the triplets contain trajs that need to be specified `base` or `novel`
        self.pred_cls_splits  = pred_cls_splits
        if assign_label is None: # None means not specified
            self.assign_label = False if "test" in self.dataset_splits else True
        else:
            self.assign_label = assign_label
        
        self.enti_cls_spilt_info = load_json(enti_cls_spilt_info_path)
        self.pred_cls_split_info = load_json(pred_cls_split_info_path)
        self.enti_cls2id = self.enti_cls_spilt_info["cls2id"] 
        self.pred_cls2id = self.pred_cls_split_info["cls2id"]
        self.enti_cls2split = self.enti_cls_spilt_info["cls2split"]
        self.pred_cls2split = self.pred_cls_split_info["cls2split"]
        self.pred_num_base = sum([v=="base" for v in self.pred_cls_split_info["cls2split"].values()])
        self.pred_num_novel = sum([v=="novel" for v in self.pred_cls_split_info["cls2split"].values()])
        # assert self.pred_num_base == 92 and self.pred_num_novel == 40


        self.dataset_dir = dataset_dir  #  train & test, # e.g., ILSVRC2015_train_00405001.json
        self.traj_info_dir = traj_info_dir                # e.g., ILSVRC2015_train_00405001-0352-0384.json
        self.traj_features_dir = traj_features_dir        # e.g., ILSVRC2015_train_00405001-0352-0384.npy
        self.traj_embd_dir = traj_embd_dir
        self.cache_dir = cache_dir
        self.traj_len_th = traj_len_th
        self.min_region_th = min_region_th
        self.vpoi_th = vpoi_th
        self.gt_training_traj_supp = False
        self.cache_tag = cache_tag
        if not os.path.exists(self.cache_dir):
            os.makedirs(cache_dir)
        
        self.segment_tags = self.prepare_segment_tags()  # len == 15146 for vidvrd-train
        self.video_annos = self.load_video_annos()
        self.det_traj_infos = self.get_traj_infos()
        if gt_training_traj_supp is not None:
            self.gt_training_traj_supp = True
            assert self.dataset_splits == ("train",)
            self.gt_traj_embd_dir = gt_training_traj_supp["embd_dir"]
            self.gt_traj_track_dir = gt_training_traj_supp["traj_dir"]
            self.gt_traj_feats_dir = gt_training_traj_supp["feature_dir"]
            self.merge_gt_traj()
        
        if self.dataset_splits == ("train",):
            self.segment_tags = self.filter_segments()  #
        
        
        if self.assign_label:
            assert self.dataset_splits == ("train",) , "we only do label assignment for train set"
            path_ = os.path.join(cache_dir,"{}VidVRDtrain_Labels_withGtTrainingData_th-{}-{}-{}.pkl".format(self.cache_tag,traj_len_th,min_region_th,vpoi_th))
            
            if os.path.exists(path_):
                print("assigned_labels loading from {}".format(path_))
                with open(path_,'rb') as f:
                    assigned_labels = pickle.load(f)
            else:
                print(f"no cache file found, assigning labels..., {path_}")
                assigned_labels = self.label_assignment()
                with open(path_,'wb') as f:
                    pickle.dump(assigned_labels,f)
                print("assigned_labels saved at {}".format(path_))
            print("len(assigned_labels) =",len(assigned_labels))
            self.segment_tags = sorted(assigned_labels.keys())  # 
            self.assigned_labels = assigned_labels
        
        # rel_pos_features is not required for label assignment, so we do this after filter
        split_tag = "".join(self.dataset_splits)
        if self.gt_training_traj_supp:
            self.cache_tag += "withGtTrainingData"
        
        path_ = os.path.join(cache_dir,"{}_VidVRD{}_rel_pos_features_th-{}-{}.pkl".format(self.cache_tag,split_tag,traj_len_th,self.min_region_th))
        if os.path.exists(path_):
            print("rel_pos_features loading from {}".format(path_))
            with open(path_,'rb') as f:
                rel_pos_features = pickle.load(f)
        else:
            print("no cache file found, ", end="")
            rel_pos_features = self.get_relative_position_feature()
            with open(path_,'wb') as f:
                pickle.dump(rel_pos_features,f)
            print("rel_pos_features saved at {}".format(path_))
        self.rel_pos_features = rel_pos_features

        # self.distillation_targets = self.get_distillation_targets() # TODO
        print("--------------- dataset constructed ---------------")
    

    def prepare_segment_tags(self):
        '''
        TODO wirte more elegant code for this func
        '''
        print("preparing segment_tags for data_splits: {}".format(self.dataset_splits),end="... ")
        video_name_to_split = dict()
        for split in ["train","test"]:
            anno_dir = os.path.join(self.dataset_dir,split)
            for filename in sorted(os.listdir(anno_dir)):
                video_name = filename.split('.')[0]  # e.g., ILSVRC2015_train_00405001.json
                video_name_to_split[video_name] = split
        
        segment_tags_all = [x.split('.')[0] for x in sorted(os.listdir(self.traj_info_dir))]  # e.g., ILSVRC2015_train_00010001-0015-0045.json
        segment_tags = []
        for seg_tag in segment_tags_all:
            video_name = seg_tag.split('-')[0] # e.g., ILSVRC2015_train_00010001-0015-0045
            split = video_name_to_split[video_name]
            if split in self.dataset_splits:
                segment_tags.append(seg_tag)
        print("total: {}".format(len(segment_tags)))

        return segment_tags


    def is_filter_out(self,h,w,traj_len):
        if traj_len < self.traj_len_th:
            return True
        
        if  h < self.min_region_th or w < self.min_region_th:
            return True
        
        return False

    def get_traj_infos(self,):
        info_str = "loading traj_infos from {} ... ".format(self.traj_info_dir)
        if self.traj_len_th > 0:
            info_str += "filter out trajs with traj_len_th = {}".format(self.traj_len_th)
        if self.min_region_th > 0:
            info_str += " min_region_th = {}".format(self.min_region_th)
        print(info_str)
        '''
        tracking_results = [
            {
                'fstarts': int(fstart),      # relative frame idx w.r.t this segment
                'score': score,             # float scalar
                'bboxes': bboxes.tolist()   # list[list], len == num_frames, format xyxy

                ### for det traj
                'label':  cls_id            # int, w.r.t VinVL classes
                
                #### for gt_traj in train-set
                'class':  class_name        # str    
                'tid':tid                   # int, original tid in annotation (w.r.t video range)
            },
            ...
        ]  # len(tracking_results) == num_tracklets
        '''

        
        
        traj_infos = dict()
        for seg_tag in tqdm(self.segment_tags):
            
            path = os.path.join(self.traj_info_dir,seg_tag + ".json")
            with open(path,'r') as f:
                tracking_results = json.load(f)

            res0 = tracking_results[0]
            # print(res0.keys())
            has_cls =  "class" in res0.keys()
            has_tid = "tid" in res0.keys()
            

            fstarts = []
            scores = []
            bboxes = []
            VinVL_clsids = []
            cls_ids = []
            tids = []
            ids_left = []
            for ii,res in enumerate(tracking_results):
                traj_len = len(res["bboxes"])
                h = max([b[3]-b[1] for b in res["bboxes"]])
                w = max([b[2]-b[0] for b in res["bboxes"]])
                if self.is_filter_out(h,w,traj_len):
                    continue
                
                fstarts.append(res["fstart"])  # relative frame_id  w.r.t segment fstart
                scores.append(res["score"])
                bboxes.append(torch.as_tensor(res["bboxes"])) # (num_boxes, 4)
                # VinVL_clsids.append(res["label"])
                ids_left.append(ii)
                if has_cls:
                    cls_ids.append(self.enti_cls2id[res["class"]]) 
                if has_tid:
                    tids.append(res["tid"])
                
            
            if ids_left:  # i.e., if ids_left != []
                ids_left = torch.as_tensor(ids_left)
                path = os.path.join(self.traj_features_dir,seg_tag+'.npy')
                traj_features = np.load(path).astype('float32')  # (n_det, 2048)
                traj_features = torch.from_numpy(traj_features)[ids_left,:]  # (n_det',2048)

                traj_info = {
                    "fstarts":torch.as_tensor(fstarts), # shape == (n_det,)
                    "scores":torch.as_tensor(scores),  # shape == (n_det,)
                    "bboxes":bboxes,  # list[tensor] , len== n_det, each shape == (num_boxes, 4)
                    "features":traj_features,     # shape == (n_det, 2048)
                    # "VinVL_clsids":torch.as_tensor(VinVL_clsids),  # shape == (n_det,)
                    "ids_left":ids_left
                }
                if has_cls:
                    traj_info.update({"cls_ids":torch.as_tensor(cls_ids)})  # len == n_det
                    # print(cls_ids)
                if has_tid:
                    traj_info.update({"tids":torch.as_tensor(tids)})
            else:
                traj_info = None

            traj_infos[seg_tag] = traj_info
        
        return traj_infos


    def merge_gt_traj(self):
        traj_dir = self.gt_traj_track_dir
        feature_dir = self.gt_traj_feats_dir

        print("merge gt traj training data ...")
        for filename in tqdm(sorted(os.listdir(traj_dir))):
            seg_tag = filename.split('.')[0]
            

            path = os.path.join(traj_dir,filename)
            with open(path,'r') as f:
                gt_results = json.load(f)
            path = os.path.join(feature_dir,seg_tag+".npy")
            gt_features = np.load(path).astype('float32')  # (num_traj, 2048)
            
            fstarts = []
            scores = []
            bboxes = []
            for res in gt_results:
                fstarts.append(res["fstart"])  # relative frame_id  w.r.t segment fstart
                scores.append(res["score"])
                bboxes.append(torch.as_tensor(res["bboxes"])) # (num_boxes, 4)
            
            if scores:  # i.e., if scores != []
                
                fstarts = torch.as_tensor(fstarts) # shape == (num_traj,)
                scores = torch.as_tensor(scores)  # shape == (num_traj,)
                features = torch.from_numpy(gt_features)  # shape == (num_traj, 2048)

                ### update
                det_traj_info = self.det_traj_infos[seg_tag]  # if det traj_info is None  ####FIXME
                det_traj_info["fstarts"] = torch.cat([det_traj_info["fstarts"],fstarts],dim=0)
                det_traj_info["scores"] = torch.cat([det_traj_info["scores"],scores],dim=0)
                det_traj_info["features"] = torch.cat([det_traj_info["features"],features],dim=0)
                det_traj_info["bboxes"] = det_traj_info["bboxes"] + bboxes
                self.det_traj_infos[seg_tag] = det_traj_info
            else:
                pass


    def __len__(self):

        return len(self.segment_tags)

    def count_pos_instances(self,labels):
        raise NotImplementedError
        # this is deprecated

        # labels  # (n_pair_aft_filter,num_pred_cats)

        if self.pred_label_split_keep == "all":
            pos_mask = torch.any(labels[:,1:].type(torch.bool),dim=-1)  # (n_pair,)
        elif self.pred_label_split_keep == "base":
            base_cls_labels = labels[:,1:self.pred_num_base+1]  # for base class
            pos_mask = torch.any(base_cls_labels.type(torch.bool),dim=-1)  # (n_pair,)
        elif self.pred_label_split_keep == "novel":
            novel_cls_labels = labels[:self.pred_num_base+1:]
            pos_mask = torch.any(novel_cls_labels.type(torch.bool),dim=-1)  # (n_pair,)
        else:
            assert False

        return pos_mask.sum()


    def filter_segments(self):
        print("filter out segments with traj_cls_splits={}, pred_cls_splits={}".format(self.traj_cls_splits,self.pred_cls_splits))
        segment_tags_have_labels = []
        for seg_tag in self.segment_tags:
            video_name, seg_fs, seg_fe = seg_tag.split('-')  # e.g., "ILSVRC2015_train_00010001-0016-0048"
            seg_fs,seg_fe = int(seg_fs),int(seg_fe)

            relations = self.video_annos[video_name]["relation_instances"]
            trajid2cls_map = {traj["tid"]:traj["category"] for traj in self.video_annos[video_name]["subject/objects"]}
            # e.g., {0: 'bicycle', 1: 'bicycle', 2: 'person', 3: 'person', 5: 'person'} # not necessarily continuous
            count = 0
            for rel in relations: # loop for relations of all segments in this video, some segments may have no annotation

                s_tid,o_tid = rel["subject_tid"],rel["object_tid"]
                s_cls,o_cls = trajid2cls_map[s_tid],trajid2cls_map[o_tid]
                s_split,o_split = self.enti_cls2split[s_cls],self.enti_cls2split[o_cls]
                p_cls = rel["predicate"]
                if not ((s_split in self.traj_cls_splits) and (o_split in self.traj_cls_splits)):
                    continue
                if not (self.pred_cls2split[p_cls] in self.pred_cls_splits):
                    continue

                fs,fe =  rel["begin_fid"],rel["end_fid"]  # [fs,fe)  fe is exclusive (from annotation)
                if not (seg_fs <= fs and fe <= seg_fe):  # we only select predicate that within this segment
                    continue
                # assert (s_tid in traj_ids) and (o_tid in traj_ids)
                count += 1
            if count == 0:
                continue
            segment_tags_have_labels.append(seg_tag)
        print("done. {} segments left".format(len(segment_tags_have_labels)))
        return segment_tags_have_labels

    def get_relative_position_feature(self):
        print("preparing relative position features ...")
        rel_pos_features = dict()
        for seg_tag in tqdm(self.segment_tags):
            ## 1. 
            # (
            #     tids_wrt_oritraj,   # (n_pair_aft_filter,2)
            #     tids_wrt_trajembds, # (n_pair_aft_filter,2)
            #     trajpair_unionembds # (n_pair_aft_filter,256)
            # ) = self.traj_pair_infos[seg_tag]
            # sids = tids_wrt_oritraj[:,0]
            # oids = tids_wrt_oritraj[:,1]

            traj_fstarts = self.det_traj_infos[seg_tag]["fstarts"]  # (n_det,)
            traj_bboxes = self.det_traj_infos[seg_tag]["bboxes"] # list[tensor] , len == n_det, each shape == (num_boxes, 4)
            n_det = len(traj_bboxes)
            pair_ids = trajid2pairid(n_det)
            sids = pair_ids[:,0]
            oids = pair_ids[:,1]

            ## 2.
            s_trajs = [traj_bboxes[idx] for idx in sids]  # format: xyxy
            o_trajs = [traj_bboxes[idx] for idx in oids]  # len == n_pair, each shape == (n_frames,4)

            s_fstarts = traj_fstarts[sids]  # (n_pair_aft_filter,)
            o_fstarts = traj_fstarts[oids]  # 

            s_lens = torch.as_tensor([x.shape[0] for x in s_trajs],device=s_fstarts.device)  # (n_pair_aft_filter,)
            o_lens = torch.as_tensor([x.shape[0] for x in o_trajs],device=o_fstarts.device)

            s_duras = torch.stack([s_fstarts,s_fstarts+s_lens],dim=-1)  # (n_pair_aft_filter,2)
            o_duras = torch.stack([o_fstarts,o_fstarts+o_lens],dim=-1)  # (n_pair_aft_filter,2)

            s_bboxes = [torch.stack((boxes[0,:],boxes[-1,:]),dim=0) for boxes in s_trajs]  # len == n_pair_aft_filter, each shape == (2,4)
            s_bboxes = torch.stack(s_bboxes,dim=0)  # (n_pair_aft_filter, 2, 4)  # 2 stands for the start & end bbox of the traj

            o_bboxes = [torch.stack((boxes[0,:],boxes[-1,:]),dim=0) for boxes in o_trajs]
            o_bboxes = torch.stack(o_bboxes,dim=0)  # (n_pair_aft_filter, 2, 4)

        
            ## 3. calculate relative position feature
            subj_x, subj_y, subj_w, subj_h = _to_xywh(s_bboxes.float())  # (n_pair_aft_filter,2)
            obj_x, obj_y, obj_w, obj_h = _to_xywh(o_bboxes.float())      # (n_pair_aft_filter,2)

            log_subj_w, log_subj_h = torch.log(subj_w), torch.log(subj_h)
            log_obj_w, log_obj_h = torch.log(obj_w), torch.log(obj_h)

            rx = (subj_x-obj_x)/obj_w   # (n_pair_aft_filter,2), 2 stands for the start & end bbox of the traj
            ry = (subj_y-obj_y)/obj_h
            rw = log_subj_w-log_obj_w
            rh = log_subj_h-log_obj_h
            ra = log_subj_w+log_subj_h-log_obj_w-log_obj_h
            rt = (s_duras-o_duras) / 30  # (n_pair_aft_filter,2)
            rel_pos_feat = torch.cat([rx,ry,rw,rh,ra,rt],dim=-1)  # (n_pair_aft_filter,12)

            rel_pos_features[seg_tag] =  rel_pos_feat

        return rel_pos_features

    def __getitem__(self,idx):
        seg_tag = self.segment_tags[idx]
        det_traj_info = deepcopy(self.det_traj_infos[seg_tag]) # (n_traj,)  including gt_trajs
        
        '''
        det_traj_info = {
            "fstart":torch.as_tensor(fstarts), # shape == (n_traj,),  n_traj == n_det + n_gt
            "scores":torch.as_tensor(scores),  # shape == (n_traj,)
            "bboxes":bboxes,  # list[tensor] , len== n_traj, each shape == (num_boxes, 4)
            "features":torch.from_numpy(traj_features)  # shape == (n_traj, 2048)
            "VinVL_clsids":torch.as_tensor(cls_ids),  # shape == (n_det',)  only for det_traj
            "ids_left":ids_left      only for det_traj
        }
        '''
        ids_left = det_traj_info["ids_left"]
        (
            traj_embds,              # (n_traj,256) 
            traj_ids_aft_filter,     # (n_det',)  only for det_traj
        ) = self.get_traj_embds(seg_tag)  # this is for distillation targets

        assert torch.all(ids_left == traj_ids_aft_filter)
        
        rel_pos_feat = deepcopy(self.rel_pos_features[seg_tag])  # (n_pair,12), n_pair = n_det*(n_det-1)
        if self.assign_label:
            labels = deepcopy(self.assigned_labels[seg_tag])  #
            pred_labels = labels["predicate"]  # (n_pair,num_pred_cats)
            so_labels = labels["entity"]  # (n_pair,2)  # NOTE currently this is not used, because we train OpenVocTrajCls separately
        else:            
            pred_labels = None
            so_labels = None
        
        return seg_tag,det_traj_info,traj_embds,traj_ids_aft_filter,rel_pos_feat,pred_labels

    
    def get_collator_func(self):
        
        def collator_func(batch_data):
            # this collator_func simply swaps the order of inner and outer of batch_data
            '''
            seg_tag,det_traj_info,traj_pair_info,rel_pos_feat,labels = batch_data[0]
            seg_tag:  e.g., "ILSVRC2015_train_00010001-0016-0048"

            det_traj_info = {
                "fstart":torch.as_tensor(fstarts), # shape == (n_det,)
                "scores":torch.as_tensor(scores),  # shape == (n_det,)
                "bboxes":bboxes,  # list[tensor] , len== n_det, each shape == (num_boxes, 4)
                "features":torch.from_numpy(traj_features)  # shape == (n_det, 2048)
            }
            rel_pos_feat,           (n_pair,12)  n_pair = n_det*(n_det-1)
            labels                  None, or (n_pair,)  # range: 0 ~ num_base (0~25)
            '''

            bsz = len(batch_data)
            num_rt_vals = len(batch_data[0])
            return_values = tuple([batch_data[bid][vid] for bid in range(bsz)] for vid in range(num_rt_vals))
            
            return return_values

        return collator_func
 

    def get_anno(self):
        raise NotImplementedError
    
    def load_video_annos(self):
        # avoid loading the same video's annotations multiple times (i.e., for multiple segments in a same video)
        annos = dict()
        for split in self.dataset_splits:
            anno_dir = os.path.join(self.dataset_dir,split)
            for filename in sorted(os.listdir(anno_dir)):
                video_name = filename.split('.')[0]  # e.g., ILSVRC2015_train_00405001.json
                path = os.path.join(anno_dir,filename)

                with open(path,'r') as f:
                    anno_per_video = json.load(f)  # refer to `datasets/vidvrd-dataset/format.py`
                annos[video_name] = anno_per_video

        return annos


    def get_traj_embds(self,seg_tag):
        # for train, we need to merge gt_traj_embds
        # for test, only use det_traj_embds
        
        path = os.path.join(self.traj_embd_dir,seg_tag+'.npy')
        temp = np.load(path).astype('float32')  # (n_traj_after_filter, 1+256)
        temp = torch.from_numpy(temp)
        traj_embds = temp[:,1:]                 # (n_traj_after_filter, 256)
        traj_ids = temp[:,0].type(torch.long)   # (n_traj_after_filter,)

        if self.gt_training_traj_supp:
            path = os.path.join(self.gt_traj_embd_dir,seg_tag+'.npy')
            temp = np.load(path).astype('float32')  # (n_gt, 1+256)
            gt_traj_embds = torch.from_numpy(temp)[:,1:] # we do not filter gt_traj w.r.t traj_len_th

            traj_embds = torch.cat([traj_embds,gt_traj_embds],dim=0)
        
        det_traj_ids_wrt_tracking = traj_ids
        
        return traj_embds,det_traj_ids_wrt_tracking


    def label_assignment(self):
        '''
        TODO FIXME write this func in a multi-process manner
        currently we use `class VidVRDUnifiedDatasetForLabelAssign` 
        and wrap it using torch's DataLoader to assign label in a multi-process manner 
        refer to  "tools/VidVRD_label_assignment.py"
        '''
        print("please use `tools/VidVRD_label_assignment.py` to pre-assign label and save as cache")
        raise NotImplementedError

        import multiprocessing

        ids_list = list(range(len(self.segment_tags)))

        p= multiprocessing.Pool(8)
        
        assigned_labels = dict()
        count = 0
        for results in (pbar := tqdm(
            p.imap(self._getitem_for_assign_label, ids_list),
            total=len(ids_list)
        )):  # RuntimeError: unable to mmap 960 bytes from file </torch_31123_868227000_64317>: Cannot allocate memory (12)
            # Why ?????
            seg_tag,assigned_pred_labels,assigned_so_labels,mask = results
            num_pos_pair = mask.sum()
            if num_pos_pair > 0:
                count+=1

            assigned_labels[seg_tag] = {
                "predicate": assigned_pred_labels,
                "entity":assigned_so_labels
            }

        p.close()
        p.join()
        
        return assigned_labels


    def _getitem_for_assign_label(self, idx):
        '''
        NOTE at this step, all segments have label, we have filtered out segments without label
        TODO use traj_cls_split = ("base",) to filter
        '''
        seg_tag = self.segment_tags[idx]
        video_name, seg_fs, seg_fe = seg_tag.split('-')  # e.g., "ILSVRC2015_train_00010001-0016-0048"
        seg_fs,seg_fe = int(seg_fs),int(seg_fe)
        gt_traj_info = self.gt_traj_infos[seg_tag]
        
        ## 1. construct gt traj pair
        gt_triplets = defaultdict(list)
        relations = self.video_annos[video_name]["relation_instances"]
        trajid2cls_map = {traj["tid"]:traj["category"] for traj in self.video_annos[video_name]["subject/objects"]}
        # e.g., {0: 'bicycle', 1: 'bicycle', 2: 'person', 3: 'person', 5: 'person'} # not necessarily continuous
        
        # after `self.filter_segments()`, each segment at least has 1 base class sampel
        # BUT can also have novel samples
        for rel in relations: 
            s_tid,o_tid = rel["subject_tid"],rel["object_tid"]
            s_cls,o_cls = trajid2cls_map[s_tid],trajid2cls_map[o_tid]
            s_split,o_split = self.enti_cls2split[s_cls],self.enti_cls2split[o_cls]
            p_cls = rel["predicate"]
            if not ((s_split in self.traj_cls_splits) and (o_split in self.traj_cls_splits)):
                continue
            if not (self.pred_cls2split[p_cls] in self.pred_cls_splits):
                continue           
            fs,fe =  rel["begin_fid"],rel["end_fid"]  # [fs,fe)  fe is exclusive (from annotation)
            if not (seg_fs <= fs and fe <= seg_fe):  # we only select predicate that within this segment
                continue
            assert seg_fs == fs and seg_fe == fe
            assert (s_tid in gt_traj_info.keys()) and (o_tid in gt_traj_info.keys()) 
            
            gt_triplets[(s_tid,o_tid)].append((s_cls,p_cls,o_cls))
        assert len(gt_triplets) > 0

        gt_s_trajs = []
        gt_o_trajs = []
        gt_s_fstarts = []
        gt_o_fstarts = []
        gt_pred_vecs = []
        gt_so_clsids = []
        for k,spo_cls_list in gt_triplets.items(): # loop for traj pair
            s_tid,o_tid = k
            pred_list = [spo_cls[1] for spo_cls in spo_cls_list]
            s_cls,o_cls = spo_cls_list[0][0],spo_cls_list[0][2]
            gt_so_clsids.append(
                [self.enti_cls2id[s_cls],self.enti_cls2id[o_cls]]
            )

            s_traj = gt_traj_info[s_tid]
            o_traj = gt_traj_info[o_tid]
            s_fs = s_traj["fstarts"]  # w.r.t the whole video
            o_fs = o_traj["fstarts"]
            s_boxes = s_traj["bboxes"]
            o_boxes = o_traj["bboxes"]
            assert len(s_boxes) == 30 and len(o_boxes) == 30 and s_fs == seg_fs and o_fs == seg_fs
            multihot = torch.zeros(size=(self.pred_num_base+self.pred_num_novel+1,))  # (num_pred_cls,) n_base+n_novel+1, including background
            for pred in pred_list:
                p_cls_id = self.pred_cls_split_info["cls2id"][pred]
                multihot[p_cls_id] = 1
            

            
            gt_s_trajs.append(s_boxes)
            gt_o_trajs.append(o_boxes)
            gt_s_fstarts.append(s_fs - seg_fs) # w.r.t the segment
            gt_o_fstarts.append(o_fs - seg_fs) 
            gt_pred_vecs.append(multihot)
        gt_s_fstarts = torch.as_tensor(gt_s_fstarts)  # (n_gt_pair,)
        gt_o_fstarts = torch.as_tensor(gt_o_fstarts)
        gt_pred_vecs = torch.stack(gt_pred_vecs,dim=0) # (n_gt_pair,num_pred_cats)  # for multi-label classification
        gt_so_clsids = torch.as_tensor(gt_so_clsids)   # (n_gt_pair,2)
        n_gt_pair = gt_pred_vecs.shape[0]

        ## 2. construct det traj pair

        det_traj_info = self.det_traj_infos[seg_tag]
        det_trajs = det_traj_info["bboxes"]    # list[tensor] , len == n_det, each shape == (num_boxes, 4)
        det_fstarts = det_traj_info["fstarts"]  # (n_det,)
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
        vpoi_s = vPoI_broadcast(det_s_trajs,gt_s_trajs,det_s_fstarts,gt_s_fstarts)  # (n_pair, n_gt_pair)  # 
        vpoi_o = vPoI_broadcast(det_o_trajs,gt_o_trajs,det_o_fstarts,gt_o_fstarts)  # (n_pair, n_gt_pair)
        vpoi_mat = torch.minimum(vpoi_s,vpoi_o)

        max_vpois,gt_pair_ids = torch.max(vpoi_mat,dim=-1)  # (n_pair,)
        # for each traj_pair, assign the gt_pair that has the max vPoI to it.
        mask = max_vpois > self.vpoi_th  # (n_pair, )
        assigned_pred_labels = gt_pred_vecs[gt_pair_ids,:]  # (n_pair,num_pred_cats)
        assigned_so_labels = gt_so_clsids[gt_pair_ids,:]    # (n_pair,2)

        assigned_pred_labels[~mask,:] = 0  # first, set background target as all-zero vectors (overwrite other multihot vectors)
        assigned_pred_labels[~mask,0] = 1  # then set these all-zero vectors as [1,0,0,...,0]  



        return seg_tag,assigned_pred_labels,assigned_so_labels,mask


class VidVRDUnifiedDatasetForLabelAssign(VidVRDUnifiedDataset):

    def __init__(self, **kargs):
        kargs.update({
            "assign_label":False
        })
        super().__init__(**kargs)

        assert self.dataset_splits == ("train",)
        self.gt_traj_infos = self.get_gt_traj_infos()  #

    def get_gt_traj_infos(self):
        
        print("preparing gt_traj_info ...")
        
        segment2traj_map = dict()
        for seg_tag in tqdm(self.segment_tags):
            video_name, fstart, fend = seg_tag.split('-')  # e.g., "ILSVRC2015_train_00010001-0016-0048"
            fstart,fend = int(fstart),int(fend)
            anno = deepcopy(self.video_annos[video_name])
            trajid2cls_map = {traj["tid"]:traj["category"] for traj in anno["subject/objects"]}
            # e.g., {0: 'bicycle', 1: 'bicycle', 2: 'person', 3: 'person', 5: 'person'} # not necessarily continuous
            # tid is w.r.t the whole video

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


            trajs_info_ = dict()
            for tid, info in trajs_info.items():
                if not info:  # i.e., if `info` is empty, we continue
                    # this can happen, because some segments have no annotation
                    continue
                trajs_info_[tid] = {
                    "class":trajid2cls_map[tid],
                    "fstarts": min(info["frame_ids"]),  # relative frame_id  w.r.t the whole video
                    "bboxes": torch.as_tensor(info["bboxes"])  # shape == (num_bbox,4)
                }


            if trajs_info_:
                segment2traj_map[seg_tag] = trajs_info_
            else:
                segment2traj_map[seg_tag] = None
        
        return segment2traj_map


    def __getitem__(self, idx):
        '''
        NOTE at this step, all segments have label, we have filtered out segments without label
        TODO use traj_cls_split = ("base",) to filter
        '''
        seg_tag = self.segment_tags[idx]
        video_name, seg_fs, seg_fe = seg_tag.split('-')  # e.g., "ILSVRC2015_train_00010001-0016-0048"
        seg_fs,seg_fe = int(seg_fs),int(seg_fe)
        gt_traj_info = self.gt_traj_infos[seg_tag]
        
        ## 1. construct gt traj pair
        gt_triplets = defaultdict(list)
        relations = self.video_annos[video_name]["relation_instances"]
        trajid2cls_map = {traj["tid"]:traj["category"] for traj in self.video_annos[video_name]["subject/objects"]}
        # e.g., {0: 'bicycle', 1: 'bicycle', 2: 'person', 3: 'person', 5: 'person'} # not necessarily continuous
        
        # after `self.filter_segments()`, each segment at least has 1 base class sampel
        # BUT can also have novel samples
        for rel in relations: 
            s_tid,o_tid = rel["subject_tid"],rel["object_tid"]
            s_cls,o_cls = trajid2cls_map[s_tid],trajid2cls_map[o_tid]
            s_split,o_split = self.enti_cls2split[s_cls],self.enti_cls2split[o_cls]
            p_cls = rel["predicate"]
            if not ((s_split in self.traj_cls_splits) and (o_split in self.traj_cls_splits)):
                continue
            if not (self.pred_cls2split[p_cls] in self.pred_cls_splits):
                continue           
            fs,fe =  rel["begin_fid"],rel["end_fid"]  # [fs,fe)  fe is exclusive (from annotation)
            if not (seg_fs <= fs and fe <= seg_fe):  # we only select predicate that within this segment
                continue
            assert seg_fs == fs and seg_fe == fe
            assert (s_tid in gt_traj_info.keys()) and (o_tid in gt_traj_info.keys()) 
            
            gt_triplets[(s_tid,o_tid)].append((s_cls,p_cls,o_cls))
        assert len(gt_triplets) > 0

        gt_s_trajs = []
        gt_o_trajs = []
        gt_s_fstarts = []
        gt_o_fstarts = []
        gt_pred_vecs = []
        gt_so_clsids = []
        for k,spo_cls_list in gt_triplets.items(): # loop for traj pair
            s_tid,o_tid = k
            pred_list = [spo_cls[1] for spo_cls in spo_cls_list]
            s_cls,o_cls = spo_cls_list[0][0],spo_cls_list[0][2]
            gt_so_clsids.append(
                [self.enti_cls2id[s_cls],self.enti_cls2id[o_cls]]
            )

            s_traj = gt_traj_info[s_tid]
            o_traj = gt_traj_info[o_tid]
            s_fs = s_traj["fstarts"]  # w.r.t the whole video
            o_fs = o_traj["fstarts"]
            s_boxes = s_traj["bboxes"]
            o_boxes = o_traj["bboxes"]
            assert len(s_boxes) == 30 and len(o_boxes) == 30 and s_fs == seg_fs and o_fs == seg_fs
            multihot = torch.zeros(size=(self.pred_num_base+self.pred_num_novel+1,))  # (num_pred_cls,) n_base+n_novel+1, including background
            for pred in pred_list:
                p_cls_id = self.pred_cls_split_info["cls2id"][pred]
                multihot[p_cls_id] = 1
            

            
            gt_s_trajs.append(s_boxes)
            gt_o_trajs.append(o_boxes)
            gt_s_fstarts.append(s_fs - seg_fs) # w.r.t the segment
            gt_o_fstarts.append(o_fs - seg_fs) 
            gt_pred_vecs.append(multihot)
        gt_s_fstarts = torch.as_tensor(gt_s_fstarts)  # (n_gt_pair,)
        gt_o_fstarts = torch.as_tensor(gt_o_fstarts)
        gt_pred_vecs = torch.stack(gt_pred_vecs,dim=0) # (n_gt_pair,num_pred_cats)  # for multi-label classification
        gt_so_clsids = torch.as_tensor(gt_so_clsids)   # (n_gt_pair,2)
        n_gt_pair = gt_pred_vecs.shape[0]

        ## 2. construct det traj pair

        det_traj_info = self.det_traj_infos[seg_tag]
        det_trajs = det_traj_info["bboxes"]    # list[tensor] , len == n_det, each shape == (num_boxes, 4)
        det_fstarts = det_traj_info["fstarts"]  # (n_det,)
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
        vpoi_s = vPoI_broadcast(det_s_trajs,gt_s_trajs,det_s_fstarts,gt_s_fstarts)  # (n_pair, n_gt_pair)  # 
        vpoi_o = vPoI_broadcast(det_o_trajs,gt_o_trajs,det_o_fstarts,gt_o_fstarts)  # (n_pair, n_gt_pair)
        vpoi_mat = torch.minimum(vpoi_s,vpoi_o)

        max_vpois,gt_pair_ids = torch.max(vpoi_mat,dim=-1)  # (n_pair,)
        # for each traj_pair, assign the gt_pair that has the max vPoI to it.
        mask = max_vpois > self.vpoi_th  # (n_pair, )
        assigned_pred_labels = gt_pred_vecs[gt_pair_ids,:]  # (n_pair,num_pred_cats)
        assigned_so_labels = gt_so_clsids[gt_pair_ids,:]    # (n_pair,2)

        assigned_pred_labels[~mask,:] = 0  # first, set background target as all-zero vectors (overwrite other multihot vectors)
        assigned_pred_labels[~mask,0] = 1  # then set these all-zero vectors as [1,0,0,...,0]  



        return seg_tag,assigned_pred_labels,assigned_so_labels,mask


class VidVRDGTDatasetForTrain_GIoU(VidVRDGTDatasetForTrain):
    def get_traj_pair_GIoU(self,seg_tag,sids,oids):

        n_pair = len(sids)

        traj_bboxes = self.traj_infos[seg_tag]["bboxes"] # list[tensor] , len == n_det, each shape == (num_boxes, 4)
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
    
    def getitem_for_seg(self,seg_tag):
        seg_tag = deepcopy(seg_tag)
        gt_triplets = deepcopy(self.gt_triplets[seg_tag])
        traj_infos = deepcopy(self.traj_infos[seg_tag])
        pair_ids,triplet_cls_ids = self.get_pred_labels(gt_triplets,traj_infos["tids"])

        sids = pair_ids[:,0]
        oids = pair_ids[:,1]

        relpos_feats = self.get_relative_position_feature(seg_tag,sids,oids)  # (n_pair,12)
        rel_giou = self.get_traj_pair_GIoU(seg_tag,sids,oids)  # (n_pair,2)

        traj_features  = traj_infos["features"]
        s_roi_feats = traj_features[sids,:]     # (n_pair,2048)
        o_roi_feats = traj_features[oids,:]     # as above

        traj_embds = self.get_traj_embds(seg_tag)
        s_embds = traj_embds[sids,:]  # (n_pair,256)
        o_embds = traj_embds[oids,:]


        return seg_tag,s_roi_feats,o_roi_feats,s_embds,o_embds,relpos_feats,rel_giou,triplet_cls_ids


class VidVRDUnifiedDataset_GIoU(VidVRDUnifiedDataset):
    def __init__(self, **kargs):
        super().__init__(**kargs)

        split_tag = "".join(self.dataset_splits)
        path_ = os.path.join(self.cache_dir,"{}_VidVRD{}_rel_giou_th-{}-{}.pkl".format(self.cache_tag,split_tag,self.traj_len_th,self.min_region_th))
        if os.path.exists(path_):
            print("rel_giou loading from {}".format(path_))
            with open(path_,'rb') as f:
                rel_gious = pickle.load(f)
        else:
            print("no cache file found, ", end="")
            rel_gious = self.get_traj_pair_GIoU()
            with open(path_,'wb') as f:
                pickle.dump(rel_gious,f)
            print("rel_giou saved at {}".format(path_))
        self.rel_gious = rel_gious


    def get_traj_pair_GIoU(self):
        all_rel_gious = dict()
        print("preparing GIoUs ...")
        for seg_tag in tqdm(self.segment_tags):
            traj_bboxes = self.det_traj_infos[seg_tag]["bboxes"] # list[tensor] , len == n_det, each shape == (num_boxes, 4)
            n_det = len(traj_bboxes)
            pair_ids = trajid2pairid(n_det)
            sids = pair_ids[:,0]
            oids = pair_ids[:,1]
            n_pair = pair_ids.shape[0]

            s_trajs = [traj_bboxes[idx] for idx in sids]  # format: xyxy
            o_trajs = [traj_bboxes[idx] for idx in oids]  # len == n_pair, each shape == (n_frames,4)

            start_s_box = torch.stack([boxes[0,:] for boxes in s_trajs],dim=0)  # (n_pair, 4)
            start_o_box = torch.stack([boxes[0,:] for boxes in o_trajs],dim=0)  # (n_pair, 4)

            end_s_box = torch.stack([boxes[-1,:] for boxes in s_trajs],dim=0)  # (n_pair, 4)
            end_o_box = torch.stack([boxes[-1,:] for boxes in o_trajs],dim=0)  # (n_pair, 4)

            start_giou = bbox_GIoU(start_s_box,start_o_box)[range(n_pair),range(n_pair)]  # (n_pair,)
            end_giou = bbox_GIoU(end_s_box,end_o_box)[range(n_pair),range(n_pair)]  # (n_pair,)
            se_giou = torch.stack([start_giou,end_giou],dim=-1)  # (n_pair,2)

            all_rel_gious[seg_tag] = se_giou
        return all_rel_gious
    
    def __getitem__(self, idx):
        seg_tag,det_traj_info,traj_embds,traj_ids_aft_filter,rel_pos,labels =  super().__getitem__(idx)
        rel_giou = deepcopy(self.rel_gious[seg_tag])
        rel_pos_feat = (rel_pos,rel_giou)

        return seg_tag,det_traj_info,traj_embds,traj_ids_aft_filter,rel_pos_feat,labels




#### added later, without deepcopy & for eval PredCls & SGCls (i.e., with gt traj bbox)

class VidVRDUnifiedDataset_ForEval(VidVRDUnifiedDataset):
    ## TODO write this dataset For Eval only, in a more elegant manner,  without deepcopy
    def __init__(self, 
        dataset_splits, 
        enti_cls_spilt_info_path="/home/gkf/project/VidVRD-OpenVoc/configs/VidVRD_class_spilt_info.json", 
        pred_cls_split_info_path="/home/gkf/project/VidVRD-OpenVoc/configs/VidVRD_pred_class_spilt_info.json", 
        dataset_dir="/home/gkf/project/VidVRD_VidOR/vidvrd-dataset", 
        traj_info_dir="/home/gkf/project/VidVRD-II/tracklets_results/VidVRD_segment30_tracking_results", 
        traj_features_dir="/home/gkf/project/scene_graph_benchmark/output/VidVRD_traj_features_seg30", 
        traj_embd_dir="/home/gkf/project/ALPRO/extract_features_output/vidvrd_seg30_TrajFeatures256", 
        cache_dir="/home/gkf/project/VidVRD-OpenVoc/datasets/cache", 
        pred_cls_splits=("base", ), 
        traj_cls_splits=("base", ), 
        traj_len_th=15, 
        min_region_th=5, 
        vpoi_th=0.9, 
        cache_tag="", 
        assign_label=None
    ):
        "/home/gkf/project/scene_graph_benchmark/output/VidVRDtest_tracking_results_gt"
        "/home/gkf/project/scene_graph_benchmark/output/VidVRDtest_gt_traj_features_seg30"
        "/home/gkf/project/ALPRO/extract_features_output/VidVRDtest_seg30_TrajFeatures256_gt"


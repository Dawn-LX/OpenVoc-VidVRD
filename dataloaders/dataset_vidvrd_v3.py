
import os
import json
from collections import defaultdict

from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

from utils.logger import LOGGER
from utils.utils_func import vIoU_broadcast,vPoI_broadcast,trajid2pairid

def load_json(filename):
    with open(filename, "r") as f:
        x = json.load(f)
    return x


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
    


def pair_demo():
    n_det = 3
    pair_trajids = torch.combinations(
        torch.as_tensor(range(n_det)),r=2,with_replacement=False
    )
    print(pair_trajids)


if __name__ == "__main__":
    pair_demo()
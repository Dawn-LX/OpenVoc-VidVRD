
import os
import json
import pickle
from collections import defaultdict
import multiprocessing
from copy import deepcopy,copy

import numpy as np
import torch
from tqdm import tqdm

from utils.utils_func import vIoU_broadcast,vPoI_broadcast,trajid2pairid,temporal_overlap
from utils.logger import LOGGER
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


def _reset_dataset_split(split):
    train = {x:"training" for x in ["train","training"]}
    val = {x:"validation" for x in ["val","validation"]}
    split_dict = {}
    for x in [train,val]:
        split_dict.update(x)
    
    return split_dict[split.lower()]

def get_single_traj_info(path):
    with open(path,'r') as f:
        tracking_results = json.load(f)

    n_traj = len(tracking_results)
    
    fstarts = []
    scores = []
    bboxes = []
    cls_ids = []

    for res in tracking_results:
        fstarts.append(res["fstart"])  # relative frame_id  w.r.t segment fstart
        scores.append(res["score"])
        bboxes.append(torch.as_tensor(res["bboxes"])) # (num_boxes, 4)
        cls_ids.append(res["label"]) # for train
    
    traj_info = {
        "fstarts":torch.as_tensor(fstarts), # shape == (n_det,)
        "scores":torch.as_tensor(scores),  # shape == (n_det,)
        "bboxes":bboxes,  # list[tensor] , len== n_det, each shape == (num_boxes, 4)
        "VinVL_clsids":torch.as_tensor(cls_ids),  # shape == (n_det,)
    }

    
    return n_traj,traj_info

class VidORTrajDataset(object):
    '''
    NOTE We apply `is_filter_out(h,w,traj_len)` directly after Seq-NMS tracking, and svae the tracking_results after filter as json
    so we do not use `is_filter_out` in this dataloader
    '''
    def __init__(self,
        class_splits,
        dataset_split,
        class_spilt_info_path = "configs/VidOR_OjbectClass_spilt_info.json",
        dataset_dir = "/home/gkf/project/VidVRD_VidOR/vidor-dataset",
        tracking_res_dir = {
            "train":"/home/gkf/project/VidVRD-II/tracklets_results/VidORtrain_tracking_results_th-15-5",
            "val":"/home/gkf/project/VidVRD-II/tracklets_results/VidORval_tracking_results"
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
        vIoU_th = 0.5,
        assign_labels = None, # None means not specified
        subset_idx_range = [0,7000],
        num_workers = 8,
    ):
        super().__init__()
        self.vIoU_th = vIoU_th
        self.num_workers = num_workers
        self.class_splits = tuple(cs.lower() for cs in class_splits)   # e.g., ("base","novel"), or ("base",) or ("novel",)
        self.dataset_split = dataset_split.lower()  
        assert self.dataset_split in ("train","val")
        if self.dataset_split == "val":
            subset_idx_range = [0,835]
        if assign_labels is None:
            assign_labels = self.dataset_split == "train"

        with open(class_spilt_info_path,'r') as f:
            self.class_split_info = json.load(f)
        self.cls2id_map = self.class_split_info["cls2id"]
        self.cls2spilt_map = self.class_split_info["cls2split"]

        self.dataset_dir = dataset_dir
        self.cache_dir  = cache_dir

        self.tracking_res_dir = tracking_res_dir[self.dataset_split] 
        # .../tracklets_results/VidORtrain_tracking_results_th-15-5/1010_8872539414/1010_8872539414-0825-0855.json
        self.traj_features_dir = traj_features_dir[self.dataset_split]
        self.traj_embds_dir = traj_embds_dir[self.dataset_split]
        

        self.anno_dir = os.path.join(self.dataset_dir,"annotation",_reset_dataset_split(self.dataset_split)) # .../vidor-dataset/training
        group_ids = os.listdir(self.anno_dir)
        video_names_all = []
        for gid in group_ids:
            filenames = os.listdir(os.path.join(self.anno_dir,gid))
            video_names_all  += [gid + "_" + filename.split(".")[0]  for filename in filenames]

        video_names_all = sorted(video_names_all)
        self.video_names_all = video_names_all
        sid,eid = subset_idx_range
        self.video_names = video_names_all[sid:eid]
        if self.dataset_split == "train" and len(self.video_names) != 7000:
            LOGGER.info("subset range [{}:{}]".format(sid,eid))
        self.segment_tags,self.segment_tags_all = self.prepare_segment_tags()
        

        
        LOGGER.info("loading tracking results... ")
        # for convenient, we save all 0:7000 video's traj_infos as cache data
        # if subset_idx_range[0] == 0 and subset_idx_range[1] == 7000:
        #     self.traj_infos,self.video2ntrajs = self.get_traj_infos_from_cache()
        # else:
        self.traj_annos = self.get_annos()  # we use this anno to calculate Recall for evaluate
        self.traj_infos,self.video2ntrajs = self.get_traj_infos()
        
        LOGGER.info("loading traj features ...")
        self.traj_features,self.traj_embeddings = self.get_traj_features()
        
        

        if self.dataset_split == "train" and assign_labels:
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
            all_labels = torch.split(all_labels,n_trajs_per_seg.tolist(),dim=0)

            self.segment_tags = [self.segment_tags_all[idx] for idx in labeled_seg_ids.tolist()]  # NOTE  seg_idx w.r.t all-7000

            self.assigned_labels = all_labels  # len == n_seg, each shape == (n_det,)
            del self.traj_annos  # for train we only use self.labels, del self.traj_annos (load `traj_annos` in `self.label_assignment()`)
        
        LOGGER.info("---------- dataset constructed, num labeled seg (i.e., len(self)) == {}".format(len(self)))

    def __len__(self):

        return len(self.segment_tags)


    def prepare_segment_tags(self):
        segment_tags_all = []
        for video_name in self.video_names_all:
            seg_filenames = sorted(os.listdir(os.path.join(self.tracking_res_dir,video_name)))
            for filename in seg_filenames:
                seg_tag = filename.split('.')[0] # e.g., 1010_8872539414-0825-0855
                segment_tags_all.append(seg_tag)


        
        segment_tags = []
        for video_name in self.video_names:
            seg_filenames = sorted(os.listdir(os.path.join(self.tracking_res_dir,video_name)))
            for filename in seg_filenames:
                seg_tag = filename.split('.')[0] # e.g., 1010_8872539414-0825-0855
                segment_tags.append(seg_tag)
        
        LOGGER.info("total videos:{} ; total segments: {}".format(len(self.video_names),len(segment_tags)))

        return segment_tags,segment_tags_all


    def get_traj_infos_from_cache(self):
        # this is deprecated
        cache_path = os.path.join(self.cache_dir,"VidORtrainAll7000_trajinfos.pkl")
        LOGGER.info("loading cached tracking results from {}".format(cache_path))
        if os.path.exists(cache_path):
            with open(cache_path,'rb') as f:
                tmp = pickle.load(f)
            traj_infos,video2ntrajs = tmp
        else:
            LOGGER.info("no cache file found, loading traj_infos and save as cache")
            traj_infos,video2ntrajs = self.get_traj_infos()
            tmp = (traj_infos,video2ntrajs)
            LOGGER.info("saving ...")
            with open(cache_path,'wb') as f:
                pickle.dump(tmp,f)
            LOGGER.info("Done.")
        return traj_infos,video2ntrajs



    def get_traj_infos_mp(self):
        # we have filter originally saved tracking results,
        # refer to func:`filter_track_res_and_feature` in `/home/gkf/project/VidVRD-II/video_object_detection/object_bbox2traj_segment.py`
        '''
        path: .../tracklets_results/VidORtrain_tracking_results_gt/1010_8872539414/1010_8872539414-0825-0855.json
        tracking_results = [
            {
                'fstarts': int(fstart),     # relative frame idx w.r.t this segment
                'score': score,             # -1 for gt
                'bboxes': bboxes.tolist()   # list[list], len == num_frames, format xyxy
                'label':  cls_id            # int, w.r.t VinVL classes
                
                #### for gt_traj in train-set
                'class':  class_name        # str    
                'tid':tid                   # int, original tid in annotation (w.r.t video range)
                
            },
            ...
        ]  # len(tracking_results) == num_tracklets
        '''



        results = dict()
        pbar = tqdm(total=len(self.segment_tags))
        pool = multiprocessing.Pool(processes=self.num_workers)
        for seg_tag in self.segment_tags:
            video_name, fstart, fend = seg_tag.split('-')  # e.g., "1010_8872539414-0825-0855"
            path = os.path.join(self.tracking_res_dir,video_name,seg_tag + ".json")

            res = pool.apply_async(
                get_single_traj_info,
                args=(path,),   # `args=(self,seg_tag)` is WRONG
                callback=lambda _: pbar.update()
            )  #  `.apply_async` returns a AsyncResult object.
            results[seg_tag] = res
        pool.close() # 关闭进程池，表示不能再往进程池中添加进程，需要在join之前调用
        pool.join() # 等待进程池中的所有进程执行完毕
        pbar.close()

        traj_infos = dict()
        video2ntrajs = defaultdict(list) # use this to get n_traj_list and split traj_features

        for seg_tag in self.segment_tags:
            video_name, fstart, fend = seg_tag.split('-')  # e.g., "1010_8872539414-0825-0855"

            res = results[seg_tag].get()

            n_traj,traj_info = res
            traj_infos[seg_tag] = traj_info
            video2ntrajs[video_name].append(n_traj)



        return traj_infos,video2ntrajs


    def get_traj_infos(self):
        # we have filter originally saved tracking results,
        # refer to func:`filter_track_res_and_feature` in `/home/gkf/project/VidVRD-II/video_object_detection/object_bbox2traj_segment.py`
        '''
        path: .../tracklets_results/VidORtrain_tracking_results_gt/1010_8872539414/1010_8872539414-0825-0855.json
        tracking_results = [
            {
                'fstarts': int(fstart),     # relative frame idx w.r.t this segment
                'score': score,             # -1 for gt
                'bboxes': bboxes.tolist()   # list[list], len == num_frames, format xyxy
                'label':  cls_id            # int, w.r.t VinVL classes
                
                #### for gt_traj in train-set
                'class':  class_name        # str    
                'tid':tid                   # int, original tid in annotation (w.r.t video range)
                
            },
            ...
        ]  # len(tracking_results) == num_tracklets
        '''


        traj_infos = dict()
        video2ntrajs = defaultdict(list) # use this to get n_traj_list and split traj_features
        for seg_tag in tqdm(self.segment_tags):
            video_name, fstart, fend = seg_tag.split('-')  # e.g., "1010_8872539414-0825-0855"
            path = os.path.join(self.tracking_res_dir,video_name,seg_tag + ".json")
            with open(path,'r') as f:
                tracking_results = json.load(f)

            n_traj = len(tracking_results)  # n_traj can be 0
            
            video2ntrajs[video_name].append(n_traj)

            
            fstarts = []
            scores = []
            bboxes = []
            cls_ids = []

            for res in tracking_results:
                fstarts.append(res["fstart"])  # relative frame_id  w.r.t segment fstart
                scores.append(res["score"])
                bboxes.append(torch.as_tensor(res["bboxes"])) # (num_boxes, 4)
                cls_ids.append(res["label"]) # for train
            
            traj_info = {
                "fstarts":torch.as_tensor(fstarts), # shape == (n_det,)
                "scores":torch.as_tensor(scores),  # shape == (n_det,)
                "bboxes":bboxes,  # list[tensor] , len== n_det, each shape == (num_boxes, 4)
                "VinVL_clsids":torch.as_tensor(cls_ids),  # shape == (n_det,)
            }

            traj_infos[seg_tag] = traj_info
        

        return traj_infos,video2ntrajs

    def get_traj_features(self):
        
        video2seg = defaultdict(list)
        for seg_tag in self.segment_tags:
            video_name, fstart, fend = seg_tag.split('-')  # e.g., "1010_8872539414-0825-0855"
            video2seg[video_name].append(seg_tag)

        #### traj RoI features (2048-d)
        # /home/gkf/project/scene_graph_benchmark/output/VidORtrain_gt_traj_features/0000_2401075277_traj_features.npy
        seg2trajfeature = dict()
        for video_name in tqdm(self.video_names,desc="load 2048-d RoI features"):
            path = os.path.join(self.traj_features_dir,video_name+"_traj_features.npy")
            traj_features = np.load(path) 
            traj_features = torch.from_numpy(traj_features).float()  # float32, # (N_traj,2048)
            n_traj_list = self.video2ntrajs[video_name]
            # print(video_name,sum(n_traj_list),traj_features.shape)
            traj_features = torch.split(traj_features,n_traj_list,dim=0)
            seg_tags = video2seg[video_name]
            assert len(traj_features) == len(seg_tags)
            seg2trajfeature.update(
                {seg_tag:traj_feat for seg_tag,traj_feat in zip(seg_tags,traj_features)}
            )
            
        
        assert len(seg2trajfeature) == len(self.segment_tags)

        #### traj Alpro-embeddings (256-d)
        # /home/gkf/project/ALPRO/extract_features_output/VidOR_TrajFeatures256_gt/0000_2401075277.pth
        # (N_traj,258) format of each line: [seg_id,tid,256d-feature] (tid w.r.t segment range (before traj_len filter), not original anno)
        seg2trajembds = dict()
        for video_name in tqdm(self.video_names,desc="load 256-d Alpro embeddings"):
            path = os.path.join(self.traj_embds_dir,video_name+".pth")
            tmp = torch.load(path)  # (N_traj,258)
            seg_ids = tmp[:,0]
            tids_aft_filter = tmp[:,1]  #this is deprecated, because we apply traj_len_th filter directly after the Seq-NMS and save the tracking results .json file
            traj_embds = tmp[:,2:]
            
            
            n_traj_list = self.video2ntrajs[video_name]
            
            ## code for assert 
            # LOGGER.info(tids_aft_filter)
            # assert len(tids_aft_filter) == sum(n_traj_list)
            # seg_ids = seg_ids.type(torch.long)
            # seg_ids,counts = torch.unique(seg_ids)
            # assert torch.all(
            #     torch.as_tensor(n_traj_list) == counts
            # )

            traj_embds = torch.split(traj_embds,n_traj_list,dim=0) # each shape: (n_traj_per_seg,256)
            seg_tags = video2seg[video_name]
            assert len(traj_embds) == len(seg_tags)
            seg2trajembds.update(
                {seg_tag: emb for seg_tag,emb in zip(seg_tags,traj_embds)}
            )
        assert len(seg2trajembds) == len(self.segment_tags)

        return seg2trajfeature,seg2trajembds


    def get_annos(self):
   
        LOGGER.info("preparing annotations for data_split: {}, class_splits: {} ".format(self.dataset_split,self.class_splits))
        

        # avoid loading the same video's annotations multiple times
        video_annos = dict()
        for video_name in tqdm(self.video_names,desc="load video annos"):  # loop for 7000

            gid,vid = video_name.split('_') 
            anno_path = os.path.join(self.anno_dir,gid,vid+".json")
            # .../vidor-dataset/annotation/training/0001/3058613626.json
            
            with open(anno_path,'r') as f:
                anno_per_video = json.load(f)  # refer to `datasets/vidvrd-dataset/format.py`
            video_annos[video_name] = anno_per_video
        


        segment2anno_map = dict()
        for seg_tag in tqdm(self.segment_tags,desc="prepare segment annos"):  
            video_name, fstart, fend = seg_tag.split('-')  # e.g., "ILSVRC2015_train_00010001-0015-0045"
            fstart,fend = int(fstart),int(fend)
            anno = video_annos[video_name]
            trajid2cls_map = {traj["tid"]:traj["category"] for traj in anno["subject/objects"]}
            # e.g., {0: 'bicycle', 1: 'bicycle', 2: 'person', 3: 'person', 5: 'person'} # not necessarily continuous

            trajs_info = {tid:defaultdict(list) for tid in trajid2cls_map.keys()}      
            annotated_len = len(anno["trajectories"])
            
            for frame_id in range(fstart,fend,1):  # 75， 105
                if frame_id >= annotated_len:  
                    # e.g., for "ILSVRC2015_train_00008005-0075-0105", annotated_len==90, and anno_per_video["frame_count"] == 130
                    break

                frame_anno = anno["trajectories"][frame_id]  # NOTE frame_anno can be [] (empty) for all `fstart` to `fend`
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
                split_ = self.cls2spilt_map[class_]
                if not (split_ in self.class_splits):
                    continue
                cls_id = self.class_split_info["cls2id"][class_]
                labels.append(cls_id)
                if cls_id > 50:
                    print(cls_id,class_,split_)
                fstarts.append(
                    min(info["frame_ids"]) - fstart  # relative frame_id  w.r.t segment fstart
                )
                bboxes.append(
                    torch.as_tensor(info["bboxes"])  # shape == (num_bbox,4)
                )
            # print(max(labels),"----------------------")
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
                

    
    def __getitem__(self,idx):
        seg_tag = deepcopy(self.segment_tags[idx])   # return seg_tag for debug
        traj_info = deepcopy(self.traj_infos[seg_tag])
        traj_feat = deepcopy(self.traj_features[seg_tag]) # (n_traj,2048)
        traj_embd = deepcopy(self.traj_embeddings[seg_tag]) # (n_traj,256)
        n_traj = traj_feat.shape[0]
        assert n_traj == traj_embd.shape[0]

        ### TODO add label assignment
        if self.dataset_split == "val":
            gt_anno = deepcopy(self.traj_annos[seg_tag])
            labels = None
        else:
            gt_anno = None
            labels = deepcopy(self.assigned_labels[idx])
            assert len(labels) == n_traj


        return seg_tag, traj_info,traj_feat,traj_embd, gt_anno, labels
    

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



class VidORTrajDataset_v2(object):
    '''
    this version do not use dict to pre-save data in memory
    '''
    def __init__(self,
        class_splits,
        dataset_split,
        class_spilt_info_path = "configs/VidOR_OjbectClass_spilt_info.json",
        dataset_dir = "/home/gkf/project/VidVRD_VidOR/vidor-dataset",
        tracking_res_dir = {
            "train":"/home/gkf/project/VidVRD-II/tracklets_results/VidORtrain_tracking_results_th-15-5",
            "val":"/home/gkf/project/VidVRD-II/tracklets_results/VidORval_tracking_results"
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
        vIoU_th = 0.5,
        assign_labels = None, # None means not specified
        subset_idx_range = [0,7000],
        num_workers = 8,
    ):
        super().__init__()
        self.vIoU_th = vIoU_th
        self.num_workers = num_workers
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
        self.cls2id_map = self.class_split_info["cls2id"]
        self.cls2spilt_map = self.class_split_info["cls2split"]

        self.dataset_dir = dataset_dir
        self.cache_dir  = cache_dir

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
        LOGGER.info("{} num videos:{} ; total segments: {}".format(txt_,len(self.video_names),len(self.segment_tags)))

        if self.dataset_split == "val":
            self.traj_annos = self.get_annos()  # we use this anno to calculate Recall for evaluate

        if self.dataset_split == "train" and assign_labels:
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
            all_labels = torch.split(all_labels,n_trajs_per_seg.tolist(),dim=0)

            self.segment_tags = [self.segment_tags_all[idx] for idx in labeled_seg_ids.tolist()]  # NOTE  seg_idx w.r.t all-7000
            # here we filter out segs with no labels and segs with n_det_traj = 0

            self.assigned_labels = all_labels  # len == n_seg, each shape == (n_det,)
            # del self.traj_annos  # for train we only use self.labels, del self.traj_annos (load `traj_annos` in `self.label_assignment()`)
        
        LOGGER.info("---------- dataset constructed, num labeled seg (i.e., len(self)) == {}".format(len(self)))

    def __len__(self):

        return len(self.segment_tags)


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
        
        video2seg = defaultdict(list)
        segment_tags_all = []
        for video_name in video_names_all:
            seg_filenames = sorted(os.listdir(os.path.join(self.tracking_res_dir,video_name)))
            for filename in seg_filenames:
                seg_tag = filename.split('.')[0] # e.g., 1010_8872539414-0825-0855
                segment_tags_all.append(seg_tag)
                video2seg[video_name].append(seg_tag)


        segment_tags = []
        for video_name in video_names:
            seg_filenames = sorted(os.listdir(os.path.join(self.tracking_res_dir,video_name)))
            for filename in seg_filenames:
                seg_tag = filename.split('.')[0] # e.g., 1010_8872539414-0825-0855
                segment_tags.append(seg_tag)
        
        seg2relidx = dict()
        for video_name, seg_tags in video2seg.items():
            for rel_idx,seg_tag in enumerate(seg_tags):
                seg2relidx[seg_tag] = rel_idx
            
        self.seg2relidx = seg2relidx
        self.video2seg = video2seg
        self.segment_tags = segment_tags
        self.video_names = video_names
        self.video_names_all = video_names_all
        self.segment_tags_all = segment_tags_all

        return segment_tags,segment_tags_all


    def get_traj_infos(self,seg_tag):
        # we have filter originally saved tracking results,
        # refer to func:`filter_track_res_and_feature` in `/home/gkf/project/VidVRD-II/video_object_detection/object_bbox2traj_segment.py`
        '''
        path: .../tracklets_results/VidORtrain_tracking_results_gt/1010_8872539414/1010_8872539414-0825-0855.json
        tracking_results = [
            {
                'fstarts': int(fstart),     # relative frame idx w.r.t this segment
                'score': score,             # -1 for gt
                'bboxes': bboxes.tolist()   # list[list], len == num_frames, format xyxy
                'label':  cls_id            # int, w.r.t VinVL classes
                
                #### for gt_traj in train-set
                'class':  class_name        # str    
                'tid':tid                   # int, original tid in annotation (w.r.t video range)
                
            },
            ...
        ]  # len(tracking_results) == num_tracklets
        '''
        
        video_name, fstart, fend = seg_tag.split('-')  # e.g., "1010_8872539414-0825-0855"
        path = os.path.join(self.tracking_res_dir,video_name,seg_tag + ".json")
        with open(path,'r') as f:
            tracking_results = json.load(f)

        n_trajs = len(tracking_results)

        fstarts = []
        scores = []
        bboxes = []
        cls_ids = []

        for res in tracking_results:
            fstarts.append(res["fstart"])  # relative frame_id  w.r.t segment fstart
            scores.append(res["score"])
            bboxes.append(torch.as_tensor(res["bboxes"])) # (num_boxes, 4)
            cls_ids.append(res["label"]) # for train
        
        traj_infos = {
            "fstarts":torch.as_tensor(fstarts), # shape == (n_det,)
            "scores":torch.as_tensor(scores),  # shape == (n_det,)
            "bboxes":bboxes,  # list[tensor] , len== n_det, each shape == (num_boxes, 4)
            "VinVL_clsids":torch.as_tensor(cls_ids),  # shape == (n_det,)
        }



        return traj_infos,n_trajs

    def get_traj_features(self,seg_tag):
        
        video_name, fstart, fend = seg_tag.split('-')  # e.g., "1010_8872539414-0825-0855"
        relative_idx = self.seg2relidx[seg_tag]

        #### traj RoI features (2048-d)
        # /home/gkf/project/scene_graph_benchmark/output/VidORtrain_gt_traj_features/0000_2401075277_traj_features.npy
        path = os.path.join(self.traj_features_dir,video_name+"_traj_features.npy")
        traj_features = np.load(path) 
        traj_features = torch.from_numpy(traj_features).float()  # float32, # (N_traj,2048)  
        ### TODO add relative seg_ids, i.e, shape == (N_traj,2049)
        # seg_rel_ids = traj_features[:,0]
        # mask = seg_rel_ids.type(torch.long) == relative_idx
        # traj_features = traj_features[mask,:]
        
        #### traj Alpro-embeddings (256-d)
        # /home/gkf/project/ALPRO/extract_features_output/VidOR_TrajFeatures256_gt/0000_2401075277.pth
        # (N_traj,258) format of each line: [seg_id,tid,256d-feature] (tid w.r.t segment range (before traj_len filter), not original anno)
        
        path = os.path.join(self.traj_embds_dir,video_name+".pth")
        tmp = torch.load(path)  # (N_traj,258)
        seg_rel_ids = tmp[:,0]
        tids_aft_filter = tmp[:,1]  #this is deprecated, because we apply traj_len_th filter directly after the Seq-NMS and save the tracking results .json file
        traj_embds = tmp[:,2:]
        mask = seg_rel_ids.type(torch.long) == relative_idx
        traj_embds = traj_embds[mask,:]

        return traj_features,traj_embds


    def get_annos(self):
   
        LOGGER.info("preparing annotations for data_split: {}, class_splits: {} ".format(self.dataset_split,self.class_splits))
        

        # avoid loading the same video's annotations multiple times
        video_annos = dict()
        for video_name in tqdm(self.video_names,desc="load video annos"):  # loop for 7000

            gid,vid = video_name.split('_') 
            anno_path = os.path.join(self.anno_dir,gid,vid+".json")
            # .../vidor-dataset/annotation/training/0001/3058613626.json
            
            with open(anno_path,'r') as f:
                anno_per_video = json.load(f)  # refer to `datasets/vidvrd-dataset/format.py`
            video_annos[video_name] = anno_per_video


        segment2anno_map = dict()
        for seg_tag in tqdm(self.segment_tags,desc="prepare segment annos"):  
            video_name, fstart, fend = seg_tag.split('-')  # e.g., "ILSVRC2015_train_00010001-0015-0045"
            fstart,fend = int(fstart),int(fend)
            anno = video_annos[video_name]
            trajid2cls_map = {traj["tid"]:traj["category"] for traj in anno["subject/objects"]}
            # e.g., {0: 'bicycle', 1: 'bicycle', 2: 'person', 3: 'person', 5: 'person'} # not necessarily continuous

            trajs_info = {tid:defaultdict(list) for tid in trajid2cls_map.keys()}      
            annotated_len = len(anno["trajectories"])
            
            for frame_id in range(fstart,fend,1):  # 75， 105
                if frame_id >= annotated_len:  
                    # e.g., for "ILSVRC2015_train_00008005-0075-0105", annotated_len==90, and anno_per_video["frame_count"] == 130
                    break

                frame_anno = anno["trajectories"][frame_id]  # NOTE frame_anno can be [] (empty) for all `fstart` to `fend`
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
                split_ = self.cls2spilt_map[class_]
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
                

    
    def __getitem__(self,idx):
        seg_tag = self.segment_tags[idx]   # return seg_tag for debug
        traj_info = self.get_traj_infos(seg_tag)
        traj_feat,traj_embd = self.get_traj_features(seg_tag)
        n_traj = traj_embd.shape[0]
        # assert n_traj == traj_embd.shape[0]

        ### TODO add label assignment
        if self.dataset_split == "val":
            gt_anno = deepcopy(self.traj_annos[seg_tag])
            labels = None
        else:
            gt_anno = None
            labels = deepcopy(self.assigned_labels[idx])
            assert len(labels) == n_traj

        return deepcopy(seg_tag), traj_info,traj_feat,traj_embd, gt_anno, labels
    

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
        ## TODO this dataloader has not been run yet

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
    ##### writing here
    def __init__(self,
        dataset_dir = "/home/gkf/project/VidVRD_VidOR/vidor-dataset",
        cache_dir = "datasets/cache",
        tracking_results_dir = "/home/gkf/project/VidVRD-II/tracklets_results/VidORtrain_tracking_results_gt",
        traj_features_dir = "/home/gkf/project/scene_graph_benchmark/output/VidORtrain_gt_traj_features",
        traj_embds_dir = "/home/gkf/project/ALPRO/extract_features_output/VidOR_TrajFeatures256_gt",
        traj_cls_split_info_path = "configs/VidOR_OjbectClass_spilt_info.json",
        pred_cls_split_info_path = "configs/VidOR_PredClass_spilt_info.json",
        pred_cls_splits = ("base",),
        traj_cls_splits = ("base",),
        bsz_wrt_pair = True,
        num_workers = 8,  # only for bsz_wrt_pair & multi-processing
    ):
        self.SEG_LEN = 30
        
        self.dataset_dir = dataset_dir
        self.cache_dir = cache_dir
        self.tracking_results_dir = tracking_results_dir
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
        self.bsz_wrt_pair = bsz_wrt_pair
        self.num_workers = num_workers


        self.video_names = sorted(os.listdir(self.tracking_results_dir))
        assert len(self.video_names) == 7000
        segment_tags = []
        for video_name in self.video_names:
            file_names = sorted(os.listdir(os.path.join(self.tracking_results_dir,video_name)))
            # /home/gkf/project/VidVRD-II/tracklets_results/VidORtrain_tracking_results_gt/1018_3949060528/1018_3949060528-0360-0390.json
            segment_tags += [x.split('.')[0] for x in file_names]
        self.segment_tags = segment_tags  #

        # NOTE we must get video2ntrajs before filter out segments
        # because traj_features are saved per video without filtering segments
        LOGGER.info("loading gt tracking results from {}... ".format(self.tracking_results_dir))
        self.traj_infos,self.video2ntrajs = self.get_traj_infos()
        LOGGER.info("loading gt traj features ...")
        self.seg2trajfeature,self.seg2trajembds = VidORTrajDataset.get_traj_features(self)

        LOGGER.info("loading gt_triplets and filter out segments with traj-split:{};pred-split:{} ...".format(self.traj_cls_splits,self.pred_cls_splits))
        self.gt_triplets = self.get_gt_triplets()  # filter out segs that have no relation annotation
        self.segment_tags = sorted(self.gt_triplets.keys())  
        LOGGER.info("Done. {} segments left".format(len(self.segment_tags)))
        # 3031 for pred_cls_splits = ("base","novel") & traj_cls_splits = ("base","novel")
        # 2981 for pred_cls_splits = ("base",) & traj_cls_splits = ("base","novel")
        # 1995 for pred_cls_splits = ("base",) & traj_cls_splits = ("base",)

        ### filter out other dicts with current segment_tags
        self.traj_infos = {tag:self.traj_infos[tag] for tag in self.segment_tags}
        self.video2ntrajs = {tag:self.video2ntrajs[tag] for tag in self.segment_tags}
        self.seg2trajfeature = {tag:self.seg2trajfeature[tag] for tag in self.segment_tags}
        self.seg2trajembds = {tag:self.seg2trajembds[tag] for tag in self.segment_tags}


        if self.bsz_wrt_pair:
            LOGGER.info("merge data from all segments ...")
            all_datas = []
            for seg_tag in tqdm(self.segment_tags):
                data = self.getitem_for_seg(seg_tag)
                # ( 
                #     seg_tag,
                #     s_roi_feats,    # (n_pair,2048)
                #     o_roi_feats,
                #     s_embds,        # (n_pair,256)
                #     o_embds,
                #     relpos_feats,   # (n_pair,12)
                #     triplet_cls_ids # list[tesor], len == n_pair, each shape == (n_preds,3)
                # ) = data
                n_pair = data[1].shape[0]
                for i in range(n_pair):
                    single_pair_data = tuple(x[i] for x in data[1:])
                    all_datas.append(single_pair_data)
            self.all_datas = all_datas


        '''Code for multi-processing, `Error: Segmentation fault (core dumped)`
        ### we now use multi-processing to speed up, refer to `test_API/test_multiprocessingPool.py`
        if self.bsz_wrt_pair:
            LOGGER.info("merge data from all segments ...")
            all_datas = []
            all_seg_data = []
            pbar = tqdm(total=len(self.segment_tags))
            pool = multiprocessing.Pool(processes=self.num_workers)
            for seg_tag in self.segment_tags:
                data = pool.apply_async(
                    self.getitem_for_seg,
                    args=(seg_tag,),   # `args=(self,seg_tag)` is WRONG
                    callback=lambda _: pbar.update()
                )  #  `.apply_async` returns a AsyncResult object.
                all_seg_data.append(data)
            pool.close() # 关闭进程池，表示不能再往进程池中添加进程，需要在join之前调用
            pool.join() # 等待进程池中的所有进程执行完毕
            pbar.close()
            
            for data in all_seg_data:
                data = data.get()
                n_pair = data[1].shape[0]
                for i in range(n_pair):
                    single_pair_data = tuple(x[i] for x in data[1:])
                    all_datas.append(single_pair_data)
    
            self.all_datas = all_datas
        '''

        LOGGER.info("------------- dataset constructed -----------, len(self) == {}".format(len(self)))
    

    def _get_cache_tag(self):
        # self.traj_cls_splits = ("base","novel")
        # self.pred_cls_splits = ("base",novel)
        
        traj_tag = [x[0].upper() for x in self.traj_cls_splits]
        traj_tag = "".join(traj_tag)
        pred_tag = [x[0].upper() for x in self.pred_cls_splits]
        pred_tag = "".join(pred_tag)

        cache_tag = "Traj_{}-Pred_{}".format(traj_tag,pred_tag)
        return cache_tag


    def load_video_annos(self):

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
        

        return video_annos
    


    def get_gt_triplets(self):
        
        video_annos = self.load_video_annos()

        gt_triplets_all = dict()
        for seg_tag in tqdm(self.segment_tags,desc="loading gt_triplets"):
            video_name, seg_fs, seg_fe = seg_tag.split('-')  # e.g., "1010_8872539414-0825-0855"
            seg_fs,seg_fe = int(seg_fs),int(seg_fe)
            anno = video_annos[video_name]
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
                if temporal_overlap((fs,fe),(seg_fs,seg_fe)) < self.SEG_LEN/2:
                    continue
                
                
                gt_triplets[(s_tid,o_tid)].append((s_cls,pred_cls,o_cls))
            if len(gt_triplets) > 0:
                gt_triplets_all[seg_tag] = gt_triplets
            
        return gt_triplets_all

    def get_traj_infos(self):
        '''
        tracking_results = [
            {
                'fstarts': int(fstart),     # relative frame idx w.r.t this segment
                'score': score,             # -1 for gt
                'bboxes': bboxes.tolist()   # list[list], len == num_frames, format xyxy
                'label':  cls_id            # int, w.r.t VinVL classes  # for detection
                
                #### for gt_traj in train-set
                'class':  class_name        # str    
                'tid':tid                   # int, original tid in annotation (w.r.t video range)
                
            },
            ...
        ]  # len(tracking_results) == num_tracklets
        
        '''
        traj_infos = dict()
        video2ntrajs = defaultdict(list)
        for seg_tag in tqdm(self.segment_tags):
            video_name, fstart, fend = seg_tag.split('-')  # e.g., "1010_8872539414-0825-0855"
            path = os.path.join(self.tracking_results_dir,video_name,seg_tag + ".json")
            # .../tracklets_results/VidORtrain_tracking_results_gt/1103_9196287263/1103_9196287263-0165-0195.json
            with open(path,'r') as f:
                gt_results = json.load(f)
            
            n_traj = len(gt_results)
            video2ntrajs[video_name].append(n_traj)
            
            fstarts = []
            scores = []
            bboxes = []
            tids = []
            clsids = []
            for res in gt_results:
                # LOGGER.info(res.keys())
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
                    "tids":torch.as_tensor(tids),        # shape == (num_traj,)
                    "clsids":torch.as_tensor(clsids)    # object category id
                }
            else:
                traj_info = None
            traj_infos[seg_tag] = traj_info
        
        return traj_infos,video2ntrajs


    def _to_xywh(self,bboxes):
        x = (bboxes[...,0] + bboxes[...,2])/2
        y = (bboxes[...,1] + bboxes[...,3])/2
        w = bboxes[...,2] - bboxes[...,0]
        h = bboxes[...,3] - bboxes[...,1]
        return x,y,w,h

    def get_pred_labels(self,gt_triplets,traj_tids):
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
    
    def __len__(self):
        if self.bsz_wrt_pair:
            return len(self.all_datas)
        else:
            return len(self.segment_tags)


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

        ''' Code for multi-processing
        gt_triplets = self.gt_triplets[seg_tag]
        traj_infos = self.traj_infos[seg_tag]

        if (not self.bsz_wrt_pair) or (self.bsz_wrt_pair and self.num_workers > 0):
            seg_tag = deepcopy(seg_tag)
            gt_triplets = deepcopy(gt_triplets)
            traj_infos = deepcopy(traj_infos)

            # must use deepcopy for multi-processing
        '''

        gt_triplets = self.gt_triplets[seg_tag]
        traj_infos = self.traj_infos[seg_tag]
        if not self.bsz_wrt_pair:
            seg_tag = deepcopy(seg_tag)
            gt_triplets = deepcopy(gt_triplets)
            traj_infos = deepcopy(traj_infos)

        pair_ids,triplet_cls_ids = self.get_pred_labels(gt_triplets,traj_infos["tids"])

        sids = pair_ids[:,0]
        oids = pair_ids[:,1]

        relpos_feats = self.get_relative_position_feature(seg_tag,sids,oids)  # (n_pair,12)

        traj_features  = self.seg2trajfeature[seg_tag]
        s_roi_feats = traj_features[sids,:]     # (n_pair,2048)
        o_roi_feats = traj_features[oids,:]     # as above

        traj_embds = self.seg2trajembds[seg_tag]
        s_embds = traj_embds[sids,:]  # (n_pair,256)
        o_embds = traj_embds[oids,:]


        return seg_tag,s_roi_feats,o_roi_feats,s_embds,o_embds,relpos_feats,triplet_cls_ids
 

    def get_relative_position_feature(self,seg_tag,sids,oids):

        traj_fstarts = self.traj_infos[seg_tag]["fstarts"]  # (n_det,)
        traj_bboxes = self.traj_infos[seg_tag]["bboxes"] # list[tensor] , len == n_det, each shape == (num_boxes, 4)
        

        ## 2.
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

    
        ## 3. calculate relative position feature
        subj_x, subj_y, subj_w, subj_h = self._to_xywh(s_bboxes.float())  # (n_pair,2)
        obj_x, obj_y, obj_w, obj_h = self._to_xywh(o_bboxes.float())      # (n_pair,2)

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

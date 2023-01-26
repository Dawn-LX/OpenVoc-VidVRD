 


import os
import json
import pickle
from collections import defaultdict

from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

from utils.utils_func import vIoU_broadcast,vPoI_broadcast,trajid2pairid

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



class VidVRDUnifiedDataset(object):
    '''
    corresponding to pipeline-3 refer to (refer to K:/NOTEsNOTEs/pic/image-20220221163112325.png)
    NOTE
    unlike `VidVRDTrajDataset`, this dataset does not construct gt_anno, 
    because we directly use VidVRD-helper (VidVRD-II) (which include dataset & gt_anno construction) to evaluate the raltion detection mAP & recall
    Therefore, we also do not set cls_split for relation ("base", "novel" or "all")
    (for training, we select the cls_slpit for label in `RelationClsModel`, for test, we set cls_split in evaluate func in VidVRD-II)
    '''
    def __init__(self,
        dataset_splits,
        enti_cls_spilt_info_path = "/home/gkf/project/VidVRD-OpenVoc/configs/VidVRD_class_spilt_info.json",
        pred_cls_split_info_path = "/home/gkf/project/VidVRD-OpenVoc/configs/VidVRD_pred_class_spilt_info.json",
        dataset_dir = "/home/gkf/project/VidVRD_VidOR/vidvrd-dataset",
        traj_info_dir = "/home/gkf/project/VidVRD-II/tracklets_results/VidVRD_segment30_tracking_results",
        traj_features_dir = "/home/gkf/project/scene_graph_benchmark/output/VidVRD_traj_features_seg30", # 2048-d RoI feature
        traj_embd_dir = "/home/gkf/project/ALPRO/extract_features_output/vidvrd_seg30_TrajFeatures256", ## 256-d
        cache_dir = "/home/gkf/project/VidVRD-OpenVoc/datasets/cache",
        gt_training_traj_supp = dict(
            traj_dir = "/home/gkf/project/scene_graph_benchmark/output/VidVRD_tracking_results_gt",
            feature_dir = "/home/gkf/project/scene_graph_benchmark/output/VidVRD_gt_traj_features_seg30",
            embd_dir = "/home/gkf/project/ALPRO/extract_features_output/vidvrd_seg30_TrajFeatures256_gt",
        ),
        pred_label_split_keep = "all",
        traj_cls_splits = ("base",),
        traj_len_th = 15,
        min_region_th = 5,
        vpoi_th = 0.9,
        cache_tag = "",
        assign_label = None, # None means not specified
        ):
        self.dataset_splits = tuple(ds.lower() for ds in dataset_splits)  # e.g., ("train","test"), or ("train",) or ("test",)
        self.pred_label_split_keep = pred_label_split_keep  # "all" (for train baseline) or "base" (for train OpenVoc)
        self.traj_cls_splits = traj_cls_splits # we do not assign traj labels here, but the triplets contain trajs that need to be specified `base` or `novel`
        if assign_label is None: # None means not specified
            self.assign_label = False if "test" in self.dataset_splits else True
        else:
            self.assign_label = assign_label
        
        self.enti_cls_spilt_info = load_json(enti_cls_spilt_info_path)
        self.pred_cls_split_info = load_json(pred_cls_split_info_path)
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
            self.segment_tags = self.filter_segments()  # len == 3031 for vidvrd-train
        
        
        if self.assign_label:
            assert self.dataset_splits == ("train",) , "we only do label assignment for train set"
            if self.min_region_th>0:
                path_ = os.path.join(cache_dir,"{}VidVRDtrain_PredLabel_withGtTrainingData_th-{}-{}-{}.pkl".format(cache_tag,traj_len_th,min_region_th,vpoi_th))
            else:
                path_ = os.path.join(cache_dir,"{}VidVRDtrain_PredLabel_withGtTrainingData_th-{}-{}.pkl".format(cache_tag,traj_len_th,vpoi_th))

            if os.path.exists(path_):
                print("assigned_labels loading from {}".format(path_))
                with open(path_,'rb') as f:
                    assigned_labels = pickle.load(f)
            else:
                print("no cache file found, assigning labels...")
                assigned_labels = self.label_assignment()
                with open(path_,'wb') as f:
                    pickle.dump(assign_label,f)
                print("assigned_labels saved at {}".format(path_))
            print("len(assigned_labels) =",len(assigned_labels))
            assigned_labels = {k:v for k,v in assigned_labels.items() if self.count_pos_instances(v) > 0}
            self.segment_tags = sorted(assigned_labels.keys())  # 
            print("filter out labels whose num_pos == 0 ..., Done. {} segments left".format(len(self.segment_tags)))
            self.assigned_labels = assigned_labels
        
        # rel_pos_features is not required for label assignment, so we do this after filter
        tag_ = "".join(self.dataset_splits)
        if self.gt_training_traj_supp:
            tag_ += "withGtTrainingData"
        
        if self.min_region_th>0:
            path_ = os.path.join(cache_dir,"VidVRD{}_rel_pos_features_th-{}-{}.pkl".format(tag_,traj_len_th,self.min_region_th))
        else:
            path_ = os.path.join(cache_dir,"VidVRD{}_rel_pos_features_th-{}.pkl".format(tag_,traj_len_th))
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
        traj_infos = dict()
        for seg_tag in tqdm(self.segment_tags):
            
            path = os.path.join(self.traj_info_dir,seg_tag + ".json")
            with open(path,'r') as f:
                tracking_results = json.load(f)

            fstarts = []
            scores = []
            bboxes = []
            cls_ids = []
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
                cls_ids.append(res["label"])
                ids_left.append(ii)
            
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
                    "VinVL_clsids":torch.as_tensor(cls_ids),  # shape == (n_det,)
                    "ids_left":ids_left
                }
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
        print("filter out segments that have no annotations ...")
        segment_tags_have_labels = []
        for seg_tag in self.segment_tags:
            video_name, seg_fs, seg_fe = seg_tag.split('-')  # e.g., "ILSVRC2015_train_00010001-0016-0048"
            seg_fs,seg_fe = int(seg_fs),int(seg_fe)

            relations = self.video_annos[video_name]["relation_instances"]
            count = 0
            for rel in relations: # loop for relations of all segments in this video, some segments may have no annotation
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


    def _to_xywh(self,bboxes):
        x = (bboxes[...,0] + bboxes[...,2])/2
        y = (bboxes[...,1] + bboxes[...,3])/2
        w = bboxes[...,2] - bboxes[...,0]
        h = bboxes[...,3] - bboxes[...,1]
        return x,y,w,h

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
            subj_x, subj_y, subj_w, subj_h = self._to_xywh(s_bboxes)  # (n_pair_aft_filter,2)
            obj_x, obj_y, obj_w, obj_h = self._to_xywh(o_bboxes)      # (n_pair_aft_filter,2)

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
            labels = deepcopy(self.assigned_labels[seg_tag])  # (n_pair,) range: 0 ~ 132 n_det is before filter
        else:            
            labels = None
        
        return seg_tag,det_traj_info,traj_embds,traj_ids_aft_filter,rel_pos_feat,labels

    
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
        currently we use `class VidVRDTripletDatasetForLabelAssign` 
        and wrap it using torch's DataLoader to assign label in a multi-process manner 
        '''
        print("please use `class VidVRDTripletDatasetForLabelAssign` to pre-assign label and save as cache")
        raise NotImplementedError
        
class VidVRDTripletDatasetForLabelAssign(VidVRDUnifiedDataset):

    def __init__(self, **kargs):
        kargs.update({
            "assign_label":False
        })
        super().__init__(**kargs)

        self.enti_cls2id = self.enti_cls_spilt_info["cls2id"]  # this not used
        self.enti_cls2split = self.enti_cls_spilt_info["cls2split"]
        assert self.dataset_splits == ("train",)
        self.gt_traj_infos = self.get_gt_traj_infos()  #

    def get_gt_traj_infos(self):
        
        print("preparing gt_traj_info for data_splits: {}, (include both base & novel entity classes)".format(self.dataset_splits))
        
        segment2traj_map = dict()
        for seg_tag in tqdm(self.segment_tags):  #Problem 这里包括 test set， 但是annos可能只有train
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
        return self.assign_label_for_pred_only(idx)

        # return self.assign_label_for_triplets(idx)

    def assign_label_for_pred_only(self,idx):
        '''
        NOTE at this step, all segments have label, we have filtered out segments without label
        TODO use traj_cls_split = ("base",) to filter
        '''
        seg_tag = self.segment_tags[idx]
        video_name, seg_fs, seg_fe = seg_tag.split('-')  # e.g., "ILSVRC2015_train_00010001-0016-0048"
        seg_fs,seg_fe = int(seg_fs),int(seg_fe)
        gt_traj_info = self.gt_traj_infos[seg_tag]
        trajid2cls_map = {traj["tid"]:traj["category"] for traj in self.video_annos[video_name]["subject/objects"]}
        # e.g., {0: 'bicycle', 1: 'bicycle', 2: 'person', 3: 'person', 5: 'person'} # not necessarily continuous

        ## 1. construct gt traj pair
        gt_triplets = defaultdict(list)
        relations = deepcopy(self.video_annos[video_name]["relation_instances"])
        
        for rel in relations: # loop for relations of all segments in this video, some segments may have no annotation
            s_tid,o_tid = rel["subject_tid"],rel["object_tid"]
            s_cls,o_cls = trajid2cls_map[s_tid],trajid2cls_map[o_tid]
            s_split,o_split = self.enti_cls2split[s_cls],self.enti_cls2split[o_cls]
            if not ((s_split in self.traj_cls_split) and (o_split in self.traj_cls_split)):
                continue
        #################### BUG FIXME NOTE Problem
        # 本来我们写的是先全部assign predicate label， 然后再把 base 的取出来。但是这样有问题：
        # 1. 在 max_vpois,gt_pair_ids = torch.max(vpoi_mat,dim=-1)， 可能max到的是 novel gt predicate，
        #    但是在最后取的时候，他就是background了；但是如果先过滤掉novel gt predicate，然年再匹配，那他max到的可能是base gt predicate
        # 2. 现在我们要把 object cls_split 在 assign relation label的时候也考虑进来，这样不统一
        # 所以最好是 新建一个 dataloaders/dataset_vidvrd_v2.py, 然后 在construct gt_triplets 的时候就过滤掉 novel entity & predicate
        # VidVRDGTDatasetForTrain 改成统一挺方便的， 在 dataset_vidvrd_v2 中主要是重写  VidVRDUnifiedDataset & VidVRDTripletDatasetForLabelAssign


            pred_cls = rel["predicate"]
            fs,fe =  rel["begin_fid"],rel["end_fid"]  # [fs,fe)  fe is exclusive (from annotation)
            if not (seg_fs <= fs and fe <= seg_fe):  # we only select predicate that within this segment
                continue
            assert seg_fs == fs and seg_fe == fe
            assert (s_tid in gt_traj_info.keys()) and (o_tid in gt_traj_info.keys()) 
            
            gt_triplets[(s_tid,o_tid)].append(pred_cls)
        assert len(gt_triplets) > 0

        gt_s_trajs = []
        gt_o_trajs = []
        gt_s_fstarts = []
        gt_o_fstarts = []
        gt_pred_vecs = []
        for k,pred_list in gt_triplets.items(): # loop for traj pair
            s_tid,o_tid = k
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
        gt_pred_vecs = torch.stack(gt_pred_vecs,dim=0)  # (n_gt_pair,num_pred_cats)  # for multi-label classification
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
        vpoi_s = vPoI_broadcast(det_s_trajs,gt_s_trajs,det_s_fstarts,gt_s_fstarts)  # (n_pair, n_gt_pair)  # This has BUG
        vpoi_o = vPoI_broadcast(det_o_trajs,gt_o_trajs,det_o_fstarts,gt_o_fstarts)  # (n_pair, n_gt_pair)
        vpoi_mat = torch.minimum(vpoi_s,vpoi_o)

        max_vpois,gt_pair_ids = torch.max(vpoi_mat,dim=-1)  # (n_pair,)
        # for each traj_pair, assign the gt_pair that has the max vPoI to it.
        mask = max_vpois > self.vpoi_th  # (n_pair, )
        assigned_pred_labels = gt_pred_vecs[gt_pair_ids,:]  # (n_pair,num_pred_cats)
        ################# TODO store the assigned entity labels by the way, according to gt_pair_ids

        assigned_pred_labels[~mask,:] = 0  # first, set background target as all-zero vectors (overwrite other multihot vectors)
        assigned_pred_labels[~mask,0] = 1  # then set these all-zero vectors as [1,0,0,...,0]  



        return seg_tag,assigned_pred_labels,mask

class VidVRDAlproUnionEmbdsForTestDataset(VidVRDUnifiedDataset):

    def __init__(self,
        dataset_splits,
        enti_cls_spilt_info_path = "/home/gkf/project/VidVRD-OpenVoc/configs/VidVRD_class_spilt_info.json",
        pred_cls_split_info_path = "/home/gkf/project/VidVRD-OpenVoc/configs/VidVRD_pred_class_spilt_info.json",
        dataset_dir = "/home/gkf/project/VidVRD_VidOR/vidvrd-dataset",
        traj_info_dir = "/home/gkf/project/VidVRD-II/tracklets_results/VidVRD_segment30_tracking_results",
        traj_features_dir = "/home/gkf/project/scene_graph_benchmark/output/VidVRD_traj_features_seg30", # 2048-d RoI feature
        trajpair_union_emb_dir = "/home/gkf/project/ALPRO/extract_features_output/vidvrd_seg30_TrajPairUnionBoxFeatures256", # 256-d
        cache_dir = "/home/gkf/project/VidVRD-OpenVoc/datasets/cache",
        traj_len_th = 15,
        min_region_th = 5,
        vpoi_th = 0.9,
        ):
        self.dataset_splits = tuple(ds.lower() for ds in dataset_splits)  # e.g., ("train","test"), or ("train",) or ("test",)

        
        self.enti_cls_spilt_info = load_json(enti_cls_spilt_info_path)
        self.pred_cls_split_info = load_json(pred_cls_split_info_path)
        self.pred_num_base = sum([v=="base" for v in self.pred_cls_split_info["cls2split"].values()])
        self.pred_num_novel = sum([v=="novel" for v in self.pred_cls_split_info["cls2split"].values()])
        assert self.pred_num_base == 92 and self.pred_num_novel == 40
        assert self.dataset_splits == ("test",)


        self.dataset_dir = dataset_dir  #  train & test, # e.g., ILSVRC2015_train_00405001.json
        self.traj_info_dir = traj_info_dir                # e.g., ILSVRC2015_train_00405001-0352-0384.json
        self.traj_features_dir = traj_features_dir        # e.g., ILSVRC2015_train_00405001-0352-0384.npy
        self.trajpair_union_emb_dir = trajpair_union_emb_dir
        self.cache_dir = cache_dir
        self.traj_len_th = traj_len_th
        self.min_region_th = min_region_th
        self.vpoi_th = vpoi_th
        self.gt_training_traj_supp = False
        
        self.segment_tags = self.prepare_segment_tags()  # len == 15146 for vidvrd-train
        self.video_annos = self.load_video_annos()
        self.det_traj_infos = self.get_traj_infos()


        # rel_pos_features is not required for label assignment, so we do this after filter
        tag_ = "".join(self.dataset_splits)
        if self.gt_training_traj_supp:
            tag_ += "withGtTrainingData"
        
        if self.min_region_th>0:
            path_ = os.path.join(cache_dir,"VidVRD{}_rel_pos_features_th-{}-{}.pkl".format(tag_,traj_len_th,self.min_region_th))
        else:
            path_ = os.path.join(cache_dir,"VidVRD{}_rel_pos_features_th-{}.pkl".format(tag_,traj_len_th))
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
    


    def get_trajpair_unionembds(self,seg_tag):
        path = os.path.join(self.trajpair_union_emb_dir,seg_tag+'.npy')
        temp = np.load(path)  # (n_pair_aft_filter, 2+2+256)  # for segment that has only 1 traj detected (thus no traj_pair)
        if temp.shape[0] == 0:
            traj_pair_infos = None
        else:
            temp = torch.from_numpy(temp.astype('float32'))
            tids_wrt_oritraj = temp[:,0:2].type(torch.long)      # (n_pair_aft_filter,2) # subset of `pari2traj_ids`, and not the same order at dim_0
            tids_wrt_trajembds = temp[:,2:4].type(torch.long)    # (n_pair_aft_filter,2)
            trajpair_unionembds = temp[:,4:]                     # (n_pair_aft_filter,256)
            traj_pair_infos = (tids_wrt_oritraj,tids_wrt_trajembds,trajpair_unionembds)

        return traj_pair_infos

    def __len__(self):

        return len(self.segment_tags)


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
        traj_pair_infos = self.get_trajpair_unionembds(seg_tag)  # this is for distillation targets
        # traj_pair_infos can be None

        # (
        #     tids_wrt_oritraj,   # (n_pair_aft_filter,2) # subset of `pari2traj_ids`, and not the same order at dim_0
        #     tids_wrt_trajembds, # (n_pair_aft_filter,2)
        #     trajpair_unionembds # (n_pair_aft_filter,256)
        # ) = traj_pair_infos

        
        rel_pos_feat = deepcopy(self.rel_pos_features[seg_tag])  # (n_pair,12), n_pair = n_det*(n_det-1)

        
        return seg_tag,det_traj_info,traj_pair_infos,rel_pos_feat


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

### ----------- with position feature (not relative position)

class VidVRDUnifiedDataset_WithPos(VidVRDUnifiedDataset):
    def __init__(self, **kargs):
        train_anno_dir = "datasets/vidvrd-dataset/train"
        test_anno_dir = "datasets/vidvrd-dataset/test"
        anno_paths =[os.path.join(train_anno_dir,filename) for filename in  os.listdir(train_anno_dir)]
        anno_paths = anno_paths + [os.path.join(test_anno_dir,filename) for filename in  os.listdir(test_anno_dir)]
        
        print("prepare video_wh infos ...")
        video_wh_infos = dict()
        for path in tqdm(anno_paths):
            with open(path,'r') as f:
                anno = json.load(f)        
            video_name = anno["video_id"]
            video_wh = (anno["width"],anno["height"])
            video_wh_infos[video_name] = video_wh


        self.video_wh_infos = video_wh_infos
        super().__init__(**kargs)  # init Father Class using bsz_wrt_pair=False
    
    def __getitem__(self, idx):
        seg_tag,det_traj_info,traj_embds,traj_ids_aft_filter,rel_pos_feat,labels =  super().__getitem__(idx)

        pos_feature = self.get_position_feature(seg_tag)

        return seg_tag,det_traj_info,traj_embds,traj_ids_aft_filter,pos_feature,labels
    
    def get_position_feature(self,seg_tag):
        video_name = seg_tag.split('-')[0] # e.g., ILSVRC2015_train_00010001-0016-0048
        video_wh = self.video_wh_infos[video_name]  # TODO 

        traj_fstarts = self.det_traj_infos[seg_tag]["fstarts"]  # (n_det,)
        traj_bboxes = self.det_traj_infos[seg_tag]["bboxes"] # list[tensor] , len == n_det, each shape == (num_boxes, 4), format: xyxy
   
        traj_lens = torch.as_tensor([x.shape[0] for x in traj_bboxes],device=traj_fstarts.device)  # (n_det,)

        traj_duras = torch.stack([traj_fstarts,traj_fstarts+traj_lens],dim=-1)  # (n_det,2)

        se_bboxes = [torch.stack((boxes[0,:],boxes[-1,:]),dim=0) for boxes in traj_bboxes]  # len == n_det, each shape == (2,4)
        se_bboxes = torch.stack(se_bboxes,dim=0)  # (n_det, 2, 4)  # 2 stands for the start & end bbox of the traj

        x,y,w,h = self._to_xywh(se_bboxes)  # (n_det,2)
        x,w = x/video_wh[0],w/video_wh[0]
        y,h = y/video_wh[1],h/video_wh[1]

        norm_pos_feature = torch.cat([x,y,w,h],dim=-1)  # (n_det,8)

        return norm_pos_feature

        

    
class VidVRDGTDatasetForTrain_WithPos(VidVRDGTDatasetForTrain):
    def __init__(self, **kargs):
        train_anno_dir = "datasets/vidvrd-dataset/train"
        test_anno_dir = "datasets/vidvrd-dataset/test"
        anno_paths =[os.path.join(train_anno_dir,filename) for filename in  os.listdir(train_anno_dir)]
        anno_paths = anno_paths + [os.path.join(test_anno_dir,filename) for filename in  os.listdir(test_anno_dir)]
        
        print("prepare video_wh infos ...")
        video_wh_infos = dict()
        for path in tqdm(anno_paths):
            with open(path,'r') as f:
                anno = json.load(f)        
            video_name = anno["video_id"]
            video_wh = (anno["width"],anno["height"])
            video_wh_infos[video_name] = video_wh


        self.video_wh_infos = video_wh_infos
        kargs["bsz_wrt_pair"] = False
        super().__init__(**kargs)  # init Father Class using bsz_wrt_pair=False

    
    def getitem_for_seg(self,seg_tag):
        seg_tag = deepcopy(seg_tag)
        gt_triplets = deepcopy(self.gt_triplets[seg_tag])
        traj_infos = deepcopy(self.traj_infos[seg_tag])
        pair_ids,triplet_cls_ids = self.get_pred_labels(gt_triplets,traj_infos["tids"])
        # pair_ids (n_pair,2)
        # triplet_cls_ids: list[tesor], len == n_pair, each shape == (n_preds,3)

        pos_feats = self.get_position_feature(seg_tag)  # (n_det,12)

        traj_features  = traj_infos["features"] # (n_det,2048)
        traj_embds = self.get_traj_embds(seg_tag) # (n_det,256)

        return seg_tag,traj_features,traj_embds,pos_feats,pair_ids,triplet_cls_ids
    
    def get_relative_position_feature(self,seg_tag,sids,oids):
        raise NotImplementedError

    def get_position_feature(self,seg_tag):
        video_name = seg_tag.split('-')[0] # e.g., ILSVRC2015_train_00010001-0016-0048
        video_wh = self.video_wh_infos[video_name]  # TODO 

        traj_fstarts = self.traj_infos[seg_tag]["fstarts"]  # (n_det,)
        traj_bboxes = self.traj_infos[seg_tag]["bboxes"] # list[tensor] , len == n_det, each shape == (num_boxes, 4), format: xyxy
   
        traj_lens = torch.as_tensor([x.shape[0] for x in traj_bboxes],device=traj_fstarts.device)  # (n_det,)

        traj_duras = torch.stack([traj_fstarts,traj_fstarts+traj_lens],dim=-1)  # (n_det,2)

        se_bboxes = [torch.stack((boxes[0,:],boxes[-1,:]),dim=0) for boxes in traj_bboxes]  # len == n_det, each shape == (2,4)
        se_bboxes = torch.stack(se_bboxes,dim=0)  # (n_det, 2, 4)  # 2 stands for the start & end bbox of the traj

        x,y,w,h = self._to_xywh(se_bboxes)  # (n_det,2)
        x,w = x/video_wh[0],w/video_wh[0]
        y,h = y/video_wh[1],h/video_wh[1]

        norm_pos_feature = torch.cat([x,y,w,h],dim=-1)  # (n_det,8)

        return norm_pos_feature



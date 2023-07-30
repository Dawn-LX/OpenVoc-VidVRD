

import json
import os
from tqdm import tqdm
import torch



# from dataloaders.dataset_vidor_v2 import VidORTrajDataset_v2,VidORTrajDataset_ForAssignLabels
from dataloaders.dataset_vidor_v3 import VidORTrajDataset,VidORTrajDataset_ForAssignLabels


def VidORTrajDataset_demo():
    '''
    NOTE we apply `is_filter_out(h,w,traj_len)` directly after Seq-NMS tracking, and svae the tracking_results after filter as json
    so we do not use `is_filter_out` in dataloader
    '''

    train_dataset_cfg = dict(
        class_splits = ("base",),
        dataset_split = "train",
        class_spilt_info_path = "configs/VidOR_OjbectClass_spilt_info_v2.json",
        dataset_dir = "data0/VidVRD_VidOR/vidor-dataset",
        tracking_res_dir = {
            "train":"data0/VidVRD-II/tracklets_results/VidORtrain_tracking_results_th-15-5",
            "val":"data0/VidVRD-II/tracklets_results/VidORval_tracking_results"
        },
        traj_features_dir={
            "train":"data0/scene_graph_benchmark/output/VidORtrain_traj_features_th-15-5",
            "val":"data0/scene_graph_benchmark/output/VidORval_traj_features_th-15-5",
        },
        traj_embds_dir = {
            "train":"data0/ALPRO/extract_features_output/VidOR_TrajFeatures256_th-15-5",
            "val":"data0/ALPRO/extract_features_output/VidORval_TrajFeatures256",
        },
        cache_dir = "datasets/cache_vidor/",
        vIoU_th = 0.5,
        subset_idx_range = [2000,3000]
    )


    train_dataset = VidORTrajDataset(**train_dataset_cfg)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        collate_fn = train_dataset.get_collator_func(),
        num_workers = 12,
        drop_last= False,
        shuffle= True,
    )

    print("start inference ...")
    for batch_data in tqdm(train_dataloader):
        # seg_tag, traj_info,traj_feat,traj_embd, gt_anno, labels = batch_data
        (
            segment_tags,
            batch_traj_infos,
            batch_traj_feats,
            bacth_traj_embds,
            batch_gt_annos,
            batch_labels
        ) = batch_data
        # batch_labels = torch.cat(batch_labels,dim=0)
        # print(batch_labels)
        
        # for x in batch_data:
        #     if isinstance(x,torch.Tensor):
        #         # print(x.shape)
        #         pass
        #     else:
        #         # print(type(x))
        #         pass
        # # break
        # pass


def VidORtrain_LabelAssignment():
    import argparse

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--sid", type=int,help="...")
    parser.add_argument("--eid", type=int,help="...")
    parser.add_argument("--num_workers", type=int,default=0)
    args = parser.parse_args()

    vIoU_th = 0.5
    subset_idx_range = [args.sid,args.eid]
    cache_dir = "datasets/cache_vidor2"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    

    train_dataset_cfg = dict(
        class_splits = ("base",),
        dataset_split = "train",
        class_spilt_info_path = "configs/VidOR_OjbectClass_spilt_info_v2.json",
        dataset_dir = "data0/VidVRD_VidOR/vidor-dataset",
        tracking_res_dir = {
            "train":"data0/VidVRD-II/tracklets_results/VidORtrainVideoLevel_tracking_results_th-15-5",
            "val":"None, we only assign label for train set"
        },
        traj_features_dir={
            "train":"data0/scene_graph_benchmark/output/VidORtrain_traj_features_th-15-5",
            "val":"not used",
        },
        traj_embds_dir = {
            "train":"data0/ALPRO/extract_features_output/VidOR_TrajFeatures256_th-15-5",
            "val":"not used",
        },
        cache_dir = cache_dir,
        vIoU_th = vIoU_th,
        assign_labels = None, # None means not specified
        subset_idx_range = subset_idx_range
    )


    train_dataset = VidORTrajDataset_ForAssignLabels(**train_dataset_cfg)
    seg2absidx = {seg_tag:idx for idx,seg_tag in enumerate(train_dataset.segment_tags_all)}
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        collate_fn = lambda x : x[0],
        num_workers = args.num_workers,
        drop_last= False,
        shuffle= False,
    )

    all_labels = []
    labeled_seg_tags = []
    n_trajs_per_seg = []
    for seg_tags,labels_per_video in tqdm(train_dataloader):
        num_seg_this_video = len(seg_tags)
        assert num_seg_this_video == labels_per_video
        for seg_tag,labels_per_seg in zip(seg_tags,labels_per_video):

            if labels_per_seg is None:
                continue
            # labels_per_seg.shape == (n_det,)
            
            labeled_seg_tags.append(seg_tag)
            n_trajs_per_seg.append(labels_per_seg.shape[0])
            all_labels.append(labels_per_seg)
    
    
    all_labels = torch.cat(all_labels,dim=0)
    labeled_seg_ids = [seg2absidx[seg_tag] for seg_tag in labeled_seg_tags]  # convert to seg_idx w.r.t all-7000 range
    labeled_seg_ids = torch.as_tensor(labeled_seg_ids)
    n_trajs_per_seg = torch.as_tensor(n_trajs_per_seg)
    

    to_save = {
        "all_labels":all_labels,
        "labeled_seg_ids":labeled_seg_ids,
        "n_trajs_per_seg":n_trajs_per_seg
    }
    s,e = subset_idx_range
    save_filename = "VidORtrain_DetTrajAssignedLabels_vIoU-{:.2f}_{}-{}.pth".format(vIoU_th,s,e)
    save_path = os.path.join(cache_dir,save_filename)
    torch.save(to_save,save_path)



def merge_assigned_labels():
    save_path = "datasets/cache_vidor/VidORtrain_DetTrajAssignedLabels_vIoU-0.50_all7000.pth"

    subset_idx_ranges = [(k*1000,(k+1)*1000) for k in range(7)]
    # print(subset_idx_ranges)
    # return
    all_labels = []
    labeled_seg_ids = []
    n_trajs_per_seg = []
    for sid,eid in subset_idx_ranges:
        load_path = "datasets/cache_vidor/VidORtrain_DetTrajAssignedLabels_vIoU-0.50_{}-{}.pth".format(sid,eid)

        labels_info = torch.load(load_path)
        all_labels.append(labels_info.pop("all_labels"))
        labeled_seg_ids.append(labels_info.pop("labeled_seg_ids"))
        n_trajs_per_seg.append(labels_info.pop("n_trajs_per_seg"))
    
    all_labels = torch.cat(all_labels,dim=0)
    labeled_seg_ids = torch.cat(labeled_seg_ids,dim=0)
    n_trajs_per_seg = torch.cat(n_trajs_per_seg,dim=0)


    to_save = {
        "all_labels":all_labels,
        "labeled_seg_ids":labeled_seg_ids,
        "n_trajs_per_seg":n_trajs_per_seg
    }
    torch.save(to_save,save_path)




 
if __name__ == "__main__":
    
    # modify_seg_tag_idx([2000,3000])
    #### vidor
    
    VidORtrain_LabelAssignment() 
   
    '''
    export PYTHONPATH=$PYTHONPATH:"/home/gaokaifeng/project/OpenVoc-VidVRD"
    python tools/VidOR_label_assignment.py --sid 0 --eid 1000 --num_workers 8

    # NOTE 对于VidOR traj label assignment， 
    考古了一下，历史情况是这样的：我们在做完 LabelAssignment 之后， 重写了dataloader，（即写了 `dataset_vidor_v3.py`)
    然后 TrajCls 的训练是基于dataset_vidor_v3的
    
    ############## 所以：
    在assign label的时候，用的是所以：dataset_vidor_v2，他是 loop w.r.t seg 的，用的是每个segment 一个json的tracking results， 
    assigned label results存的是所有seg的结果合在一起存为一个tensor的

    然后训练traj cls的时候（dataset_vidor_v3），用的是每个segment tracking results 合并之后的，即每个video 一个tracking results .json。
    训练traj cls的时候，batch size 是w.r.t video, 然后每个video 都会采样几个segment。
    
    ##############
    现在camera ready之后，我们只release 每个video 一个tracking results .json。
    然后VidORtrain_LabelAssignment的代码就要改一下。
    
    '''
    # VidORGTDatasetForTrain_demo()
    # demo_unique()

    
    # frame_range_stat()
    # map_clsid_to_v2()
    
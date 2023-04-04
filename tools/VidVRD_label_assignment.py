

import argparse
import os
from tqdm import tqdm
import pickle
import torch


from dataloaders.dataset_vidvrd_v2 import VidVRDUnifiedDatasetForLabelAssign



def VidVRDUnifiedDatasetForLabelAssign_assign_label(
    traj_len_th,min_region_th,vpoi_th,cache_tag,is_save=True,pred_cls_splits = ("base",),traj_cls_splits = ("base",)
):
    
    train_dataset_cfg = dict(
        dataset_splits = ("train",),
        enti_cls_spilt_info_path = "configs/VidVRD_class_spilt_info.json",
        pred_cls_split_info_path = "configs/VidVRD_pred_class_spilt_info_v2.json",
        dataset_dir = "data0/VidVRD_VidOR/vidvrd-dataset",
        traj_info_dir = "data0/VidVRD-II/tracklets_results/VidVRD_segment30_tracking_results",
        traj_features_dir = "data0/scene_graph_benchmark/output/VidVRD_traj_features_seg30", # 2048-d RoI feature
        traj_embd_dir = "data0/ALPRO/extract_features_output/vidvrd_seg30_TrajFeatures256", ## 256-d
        cache_dir = "datasets/cache_vidvrd",
        gt_training_traj_supp = dict(
            traj_dir = "data0/scene_graph_benchmark/output/VidVRD_tracking_results_gt",
            feature_dir = "data0/scene_graph_benchmark/output/VidVRD_gt_traj_features_seg30",
            embd_dir = "data0/ALPRO/extract_features_output/vidvrd_seg30_TrajFeatures256_gt",
        ),
        pred_cls_splits = ("base","novel"), # only used for train
        traj_cls_splits = ("base","novel"), # only used for train
        traj_len_th = 15,
        min_region_th = 5,
        vpoi_th = 0.9,
        cache_tag = "PredSplit_v2_FullySupervise"
    )

    dataset = VidVRDUnifiedDatasetForLabelAssign(**train_dataset_cfg)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn = lambda x :x  ,
        num_workers = 16,
        drop_last=False,
        shuffle=False,  
    )
    cache_dir = train_dataset_cfg["cache_dir"]
    # save_path = os.path.join(save_dir,"VidVRDtrain_PredLabel_withGtTrainingData_th{}.pkl".format(vpoi_th))
    save_path = os.path.join(cache_dir,"{}VidVRDtrain_Labels_withGtTrainingData_th-{}-{}-{}.pkl".format(cache_tag,traj_len_th,min_region_th,vpoi_th))

    print("start assigning label with vPoI_th={}".format(vpoi_th))
    assigned_labels = dict()
    count = 0
    count_p = 0
    for idx,batch_data in enumerate(tqdm(dataloader)):
        seg_tag,assigned_pred_labels,assigned_so_labels,mask = batch_data[0]

        num_pos_pair = mask.sum()
        if num_pos_pair > 0:
            count+=1
        
        assigned_labels[seg_tag] = {
            "predicate": assigned_pred_labels,
            "entity":assigned_so_labels
        }

    print("postive seg count = {},count_p={}".format(count,count_p))
        
    if is_save:
        with open(save_path,'wb') as f:
            pickle.dump(assigned_labels,f)
        print("result saved at: {}".format(save_path))
    



if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(description="Object Detection Demo")

    parser.add_argument("--traj_len_th", type=int,help="...")
    parser.add_argument("--min_region_th", type=int,help="...")
    parser.add_argument("--vpoi_th", type=float,help="...")
    parser.add_argument("--cache_tag", type=str,default="",help="...")
    parser.add_argument("--is_save", action="store_true",default=False)
    args = parser.parse_args()

    
    traj_len_th = args.traj_len_th
    min_region_th = args.min_region_th
    vpoi_th = args.vpoi_th
    cache_tag = args.cache_tag
    assert isinstance(vpoi_th,float)
    
    # VidVRDUnifiedDatasetForLabelAssign_assign_label(traj_len_th,min_region_th,vpoi_th,cache_tag,is_save=True)

    # for traj-base & pred-base
    VidVRDUnifiedDatasetForLabelAssign_assign_label(
        traj_len_th,
        min_region_th,
        vpoi_th,
        cache_tag,
        is_save=True,
        pred_cls_splits=("base",),
        traj_cls_splits = ("base",)
    )


    # for fully supervised
    # VidVRDUnifiedDatasetForLabelAssign_assign_label(
    #     traj_len_th,
    #     min_region_th,
    #     vpoi_th,
    #     cache_tag,
    #     is_save=True,
    #     pred_cls_splits=("base","novel"),
    #     traj_cls_splits = ("base","novel")
    # )
    '''
    python tools/VidVRD_label_assignment.py \
        --traj_len_th 15 \
        --min_region_th 5 \
        --vpoi_th 0.9 \
        --cache_tag PredSplit_v2_TrajBasePredBase \
        --is_save
    
    python tools/VidVRD_label_assignment.py \
        --traj_len_th 15 \
        --min_region_th 5 \
        --vpoi_th 0.9 \
        --cache_tag PredSplit_v2_FullySupervise \
        --is_save
    '''
    


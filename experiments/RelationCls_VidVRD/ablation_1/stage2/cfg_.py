model_pred_cfg = dict(
    dim_emb = 256,
    dim_feat = 2048,  # bbox roi feature, refer to Faster-RCNN
    dim_hidden = 768,
    num_base = 71,
    num_novel = 61,
    temperature_init =  0.02125491015613079, # learned by Alpro
    n_context_tokens = 10,
    prompter_ckpt_path = "experiments/RelationCls_VidVRD/ablation_1/stage1/model_bsz32_best_mAP-EP27.pth",
    pred_cls_split_info_path = "configs/VidVRD_pred_class_spilt_info_v2.json",
)

model_traj_cfg = dict(
    dim_roi = 2048,  # bbox roi feature, refer to Faster-RCNN
    dim_emb = 256,
    dim_hidden = 1024,
    num_base = 25,
    num_novel = 10,
    text_emb_path = "data0/VidVRD-OpenVoc/prepared_data/vidvrd_ObjTextEmbeddings.pth",
    temperature_init = 0.02125491015613079, # learned by Alpro
    loss_factor = dict(
        pos_cls = 1.0,
        neg_cls = 1.0,
        distillation = 5.0,
    )
)

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
    pred_cls_splits = ("base",), # only used for train
    traj_cls_splits = ("base",), # only used for train
    traj_len_th = 15,
    min_region_th = 5,
    vpoi_th = 0.9,
    cache_tag = "PredSplit_v2_TrajBasePredBase"
)

eval_dataset_cfg = dict(
    dataset_splits = ("test",),
    enti_cls_spilt_info_path = "configs/VidVRD_class_spilt_info.json",
    pred_cls_split_info_path = "configs/VidVRD_pred_class_spilt_info_v2.json",
    dataset_dir = "data0/VidVRD_VidOR/vidvrd-dataset",
    traj_info_dir = "data0/VidVRD-II/tracklets_results/VidVRD_segment30_tracking_results",
    traj_features_dir = "data0/scene_graph_benchmark/output/VidVRD_traj_features_seg30", # 2048-d RoI feature
    traj_embd_dir = "data0/ALPRO/extract_features_output/vidvrd_seg30_TrajFeatures256", ## 256-d
    cache_dir = "datasets/cache_vidvrd",
    gt_training_traj_supp = None,
    traj_len_th = 15,
    min_region_th = 5,
)

GTeval_dataset_cfg = dict(
    dataset_splits = ("test",),
    enti_cls_spilt_info_path = "configs/VidVRD_class_spilt_info.json",
    pred_cls_split_info_path = "configs/VidVRD_pred_class_spilt_info_v2.json",
    dataset_dir = "data0/VidVRD_VidOR/vidvrd-dataset",
    traj_info_dir = "data0/scene_graph_benchmark/output/VidVRDtest_tracking_results_gt",
    traj_features_dir = "data0/scene_graph_benchmark/output/VidVRDtest_gt_traj_features_seg30", # 2048-d RoI feature
    traj_embd_dir = "data0/ALPRO/extract_features_output/VidVRDtest_seg30_TrajFeatures256_gt", ## 256-d
    cache_dir = "datasets/cache_vidvrd",
    gt_training_traj_supp = None,
    traj_len_th = -1,
    min_region_th = -1,
    cache_tag = "gtbbox"
)

train_cfg = dict(
    batch_size          = 32,
    total_epoch         = 50,
    initial_lr          = 1e-4,
    lr_decay            = 0.2,
    epoch_lr_milestones = [30],
)


eval_cfg = dict(
    pred_topk = 10,
    return_triplets_topk = 200,
)

eval_cfg_for_train = dict(
    pred_topk = 10,
    return_triplets_topk = 200,
    ckpt_path_traj = "experiments/TrajCls_VidVRD/NoBgEmb/model_final_with_distil_w5bs128_epoch_50.pth"
)

association_cfg = dict(
    inference_topk = eval_cfg["return_triplets_topk"], ## we do this to keep compatible with VidVRD-II's code
    association_algorithm = "greedy",
    association_linkage_threshold = 0.5,
    association_nms = 1.0,
    association_topk = 200,
    association_n_workers = 12
)
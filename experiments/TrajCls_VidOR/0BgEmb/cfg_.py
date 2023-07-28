model_cfg = dict(
    dim_roi = 2048,  # bbox roi feature, refer to Faster-RCNN
    dim_emb = 256,
    dim_hidden = 1024,
    num_base = 50,
    num_novel = 30,
    text_emb_path = "data0/VidVRD-OpenVoc/prepared_data/vidor_ObjectTextEmbeddings_v2.pth",
    temperature_init = 0.02125491015613079, # learned by Alpro
    loss_factor = dict(
        pos_cls = 1.0,
        neg_cls = 1.0,
        distillation = 5.0,
    )
)

train_dataset_cfg = dict(
    class_splits = ("base",),
    dataset_split = "train",
    class_spilt_info_path = "configs/VidOR_OjbectClass_spilt_info_v2.json",
    dataset_dir = "data0/VidVRD_VidOR/vidor-dataset",
    tracking_res_dir = {
        "train":"data0/VidVRD-II/tracklets_results/VidORtrainVideoLevel_tracking_results_th-15-5",
        "val":"data0/VidVRD-II/tracklets_results/VidORvalVideoLevel_tracking_results_th-15-5"
    },
    traj_features_dir={
        "train":"data0/scene_graph_benchmark/output/VidORtrain_traj_features_th-15-5",
        "val":"data0/scene_graph_benchmark/output/VidORval_traj_features_th-15-5",
    },
    traj_embds_dir = {
        "train":"data0/ALPRO/extract_features_output/VidOR_TrajFeatures256_th-15-5",
        "val":"data0/ALPRO/extract_features_output/VidORval_TrajFeatures256",
    },
    num_sample_segs = 32,
    cache_dir = "datasets/cache_vidor",
    vIoU_th = 0.5,
    subset_idx_range = [0,7000]
)

eval_dataset_cfg = dict(
    class_splits = ("novel",),
    dataset_split = "val",
    class_spilt_info_path = "configs/VidOR_OjbectClass_spilt_info_v2.json",
    dataset_dir = "data0/VidVRD_VidOR/vidor-dataset",
    tracking_res_dir = {
        "train":"data0/VidVRD-II/tracklets_results/VidORtrainVideoLevel_tracking_results_th-15-5",
        "val":"data0/VidVRD-II/tracklets_results/VidORvalVideoLevel_tracking_results_th-15-5"
    },
    traj_features_dir={
        "train":"data0/scene_graph_benchmark/output/VidORtrain_traj_features_th-15-5",
        "val":"data0/scene_graph_benchmark/output/VidORval_traj_features_th-15-5",
    },
    traj_embds_dir = {
        "train":"data0/ALPRO/extract_features_output/VidOR_TrajFeatures256_th-15-5",
        "val":"data0/ALPRO/extract_features_output/VidORval_TrajFeatures256",
    },
)

train_cfg = dict(
    batch_size          = 16,
    total_epoch         = 50,
    initial_lr          = 1e-4,
    lr_decay            = 0.2,
    epoch_lr_milestones = [30],
)


eval_cfg = dict(
    vIoU_th = 0.5,
    batch_size = 16,
)
model_cfg = dict(
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
    class_splits = ("base",),
    dataset_split = "train",
    class_spilt_info_path = "configs/VidVRD_class_spilt_info.json",
    dataset_dir = "data0/VidVRD_VidOR/vidvrd-dataset",
    tracking_res_dir = "data0/VidVRD-II/tracklets_results/VidVRD_segment30_tracking_results_th-15-5",
    traj_features_dir = "data0/scene_graph_benchmark/output/VidVRD_traj_features_seg30_th-15-5",
    traj_embeddings_dir = "data0/ALPRO/extract_features_output/vidvrd_seg30_TrajFeatures256", # this has been filtered as th-15-5, refer to /home/gkf/project/ALPRO
    vIoU_th = 0.5,
    cache_dir = "datasets/cache_vidvrd"
)

eval_dataset_cfg = dict(
    class_splits = ("novel",),
    dataset_split = "test",
    class_spilt_info_path = "configs/VidVRD_class_spilt_info.json",
    dataset_dir = "data0/VidVRD_VidOR/vidvrd-dataset",
    tracking_res_dir = "data0/VidVRD-II/tracklets_results/VidVRD_segment30_tracking_results_th-15-5",
    traj_features_dir = "data0/scene_graph_benchmark/output/VidVRD_traj_features_seg30_th-15-5",
    traj_embeddings_dir = "data0/ALPRO/extract_features_output/vidvrd_seg30_TrajFeatures256",
    vIoU_th = 0.5,
    cache_dir = "datasets/cache_vidvrd"
)

train_cfg = dict(
    batch_size          = 128,
    total_epoch         = 50,
    initial_lr          = 1e-4,
    lr_decay            = 0.2,
    epoch_lr_milestones = [30],
)


eval_cfg = dict(
    vIoU_th = 0.5,
    batch_size = 16,
)
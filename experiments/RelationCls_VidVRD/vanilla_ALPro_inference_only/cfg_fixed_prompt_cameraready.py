model_cfg = dict(
    dim_roi = 2048,  # bbox roi feature, refer to Faster-RCNN
    dim_emb = 256,
    dim_hidden = 1024,
    num_base = 25,
    num_novel = 10,
    text_emb_path = "prepared_data/vidvrd_ObjTextEmbeddings.pth",
    temperature_init = 0.02125491015613079, # learned by Alpro
    loss_factor = dict(
        pos_cls = 1.0,
        neg_cls = 1.0,
        distillation = 5.0,
    )
)


model_pred_cfg = dict(
    num_base = 71,
    num_novel = 61,
    temperature_init =  0.02125491015613079, # learned by Alpro
    pred_cls_split_info_path = "configs/VidVRD_pred_class_spilt_info_v2.json",
    prompt_type = "separate",
    subj_prompt_template = "A video of a person or object {} something",
    obj_prompt_template =  "A video of something {} a person or object",
)

################### this is modified, for `from models.TrajClsModel_v2 import OpenVocTrajCls as OpenVocTrajCls_NoBgEmb`
model_traj_cfg = dict(
    dim_roi = 2048,  # bbox roi feature, refer to Faster-RCNN
    dim_emb = 256,
    dim_hidden = 1024,
    num_base = 25,
    num_novel = 10,
    text_emb_path = "/home/gkf/project/VidVRD-OpenVoc/prepared_data/vidvrd_ObjTextEmbeddings.pth",
    temperature_init = 0.02125491015613079, # learned by Alpro
    loss_factor = dict(
        pos_cls = 1.0,
        neg_cls = 1.0,
        distillation = 5.0,
    )
)
################### 

eval_dataset_cfg = dict(
    dataset_splits = ("test",),
    enti_cls_spilt_info_path = "configs/VidVRD_class_spilt_info.json",
    pred_cls_split_info_path = "configs/VidVRD_pred_class_spilt_info_v2.json",
    dataset_dir = "/home/gkf/project/VidVRD_VidOR/vidvrd-dataset",
    traj_info_dir = "/home/gkf/project/VidVRD-II/tracklets_results/VidVRD_segment30_tracking_results",
    traj_features_dir = "/home/gkf/project/scene_graph_benchmark/output/VidVRD_traj_features_seg30", # 2048-d RoI feature
    traj_embd_dir = "/home/gkf/project/ALPRO/extract_features_output/vidvrd_seg30_TrajFeatures256", ## 256-d
    cache_dir = "/home/gkf/project/VidVRD-OpenVoc/datasets/cache",
    gt_training_traj_supp = None,
    traj_len_th = 15,
    min_region_th = 5,
)

GTeval_dataset_cfg = dict(
    dataset_splits = ("test",),
    enti_cls_spilt_info_path = "configs/VidVRD_class_spilt_info.json",
    pred_cls_split_info_path = "configs/VidVRD_pred_class_spilt_info_v2.json",
    dataset_dir = "/home/gkf/project/VidVRD_VidOR/vidvrd-dataset",
    traj_info_dir = "/home/gkf/project/scene_graph_benchmark/output/VidVRDtest_tracking_results_gt",
    traj_features_dir = "/home/gkf/project/scene_graph_benchmark/output/VidVRDtest_gt_traj_features_seg30", # 2048-d RoI feature
    traj_embd_dir = "/home/gkf/project/ALPRO/extract_features_output/VidVRDtest_seg30_TrajFeatures256_gt", ## 256-d
    cache_dir = "/home/gkf/project/VidVRD-OpenVoc/datasets/cache",
    gt_training_traj_supp = None,
    traj_len_th = -1,
    min_region_th = -1,
    cache_tag = "gtbbox"
)



eval_cfg = dict(
    pred_topk = 10,
    return_triplets_topk = 200,
)

eval_cfg_for_train = dict(
    pred_topk = 10,
    return_triplets_topk = 200,
    ckpt_path_traj = "/home/gkf/project/VidVRD-OpenVoc/experiments/ALPro_teacher/model_OpenVoc_w15BS128_epoch_50.pth"
)

association_cfg = dict(
    inference_topk = eval_cfg["return_triplets_topk"], ## we do this to keep compatible with VidVRD-II's code
    association_algorithm = "greedy",
    association_linkage_threshold = 0.5,
    association_nms = 1.0,
    association_topk = 200,
    association_n_workers = 12
)
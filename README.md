Official code for our ICLR2023 paper: "Compositional Prompt Tuning with Motion Cues for Open-Vocabulary Video Relation Detection"
[openreview link](https://openreview.net/pdf?id=mE91GkXYipg) 

# The code is still preparing...

# data download summarize:

VidVRD:
- traj bbox 
    gt: `/home/gkf/project/scene_graph_benchmark/output/VidVRDtest_tracking_results_gt` (12M)
    det: `/home/gkf/project/VidVRD-II/tracklets_results/VidVRD_segment30_tracking_results` (1.1G)
- traj RoI features
    gt: `/home/gkf/project/scene_graph_benchmark/output/VidVRDtest_gt_traj_features_seg30` (72M)
    det: `/home/gkf/project/scene_graph_benchmark/output/VidVRD_traj_features_seg30` (6.3G)
- traj embds
    gt: `/home/gkf/project/ALPRO/extract_features_output/VidVRDtest_seg30_TrajFeatures256_gt` (14M)
    det: `/home/gkf/project/ALPRO/extract_features_output/vidvrd_seg30_TrajFeatures256` (453M)

- TrajCLs model weight: (40M each)
    old (in paper for review): `/home/gkf/project/VidVRD-OpenVoc/experiments/ALPro_teacher/model_OpenVoc_w15BS128_epoch_50.pth` 
    NoBG-wDistil (for cameready) : `/home/gkf/project/VidVRD-OpenVoc/experiments_vidvrd_trajcls/OpenVocTrajCls_NoBgEmb/model_final_with_distil_w5bs128_epoch_50.pth`

    other weights for ablations in table-1:
    `experiments_vidvrd_trajcls/OpenVocTrajCls_NoBgEmb/model_final_wo_distil_bs128_epoch_50.pth`
    `experiments_vidvrd_trajcls/OpenVocTrajCls_0BgEmb/model_final_with_distil_w5bs128_epoch_50.pth`
    `experiments_vidvrd_trajcls/OpenVocTrajCls_0BgEmb/model_final_wo_distil_bs128_epoch_50.pth`

- RelationCls model weight:

scp -r -P 3824 -o "ProxyJump gaokaifeng@10.214.160.111 -p 4637" gkf@10.214.223.101:/home/gkf/project/VidVRD-OpenVoc-release/vidvrd_traj_box_det.zip .

-q 不显示指令执行过程。
zip -r vidvrd_traj_box_det.zip /home/gkf/project/VidVRD-II/tracklets_results/VidVRD_segment30_tracking_results
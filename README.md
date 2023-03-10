Official code for our ICLR2023 paper: "Compositional Prompt Tuning with Motion Cues for Open-Vocabulary Video Relation Detection"
[openreview link](https://openreview.net/pdf?id=mE91GkXYipg) 

# The code is still preparing .....

# Requirements

- Python == 3.7 or later, Pytorch == 1.7 or later
- transformers == 4.11.3 (new version might require some modifications for ALpro's code, but also works， refer to Line 872 in `Alpro_modeling/xbert.py`) 
- for other basic packages, just run the project and download whatever needed.

# data-release summarize

**Overview**: There are 3 types of data
- tacklet bbox (or traj bbox): bounding box sequence after object tracking. 
    - Here we use Seq-NMS to perform tracking, and we do tracking in each video segment (30 frame per seg)
- traj RoI features: 2048-d RoI features obtained by FasterRCNN's RoI-Align. Here we use [VinVL](https://github.com/pzzhang/VinVL) (its model structure is FasterRCNN). 
    - We extracted the RoI feature for each tracklet (i.e., each bbox in the bbox sequence), and averaged the feature along the time axis (the released data is after this averaging).
- traj embds: 256-d embeddings obtained by the video-language pre-train model [ALpro](https://github.com/salesforce/ALPRO). 
    - **NOTE**: we call this as embedding (in README file and our code) to distinguish it with the 2048-d RoI feature.
    - Also NOTE that this is not extracted per-box of the tracklet and averaged along time axis. ALPro takes as input the video segment and output the 256-d embedding directly. In our implementation, we crop the video region according to the traj bboxes, and take this as the input to ALpro.

For each type of the above data, it includes `gt` and `det`, i.e., ground-truth traj bboxes and detection traj bboxes, with their features/embds. (certainly, we don't need Seq-NMS to perform tracking for `gt`)

## VidVRD
### Pre-prepared traj data ([MEGA cloud link](https://mega.nz/folder/AYBkxCaI#QCqV3cnIdY_9DXGUnCtSvA))

In detail, there are the following files: (where `data0/` refers to `/home/gkf/project/`)

- traj bbox 
    - gt: `vidvrd_traj_box_gt.zip`, corresponding to `data0/scene_graph_benchmark/output/VidVRDtest_tracking_results_gt`
    - det:  `vidvrd_traj_box_det.zip`, c.t. `data0/VidVRD-II/tracklets_results/VidVRD_segment30_tracking_results`
    - det-th-15-5:  `vidvrd_traj_box_det_th-15-5.zip`, c.t. `data0/VidVRD-OpenVoc/vidvrd_traj_box_det_th-15-5.zip`
        - this is used for TrajCls module only, it can be obtained by filter out trajs with length < 15 and area < 5, but we also provide this data to make sure.
- traj RoI features (2048-d)
    - gt: `vidvrd_traj_roi_gt.zip`, c.t. `data0/scene_graph_benchmark/output/VidVRDtest_gt_traj_features_seg30`
    - det:  `vidvrd_traj_roi_det.zip`, c.t. `data0/scene_graph_benchmark/output/VidVRD_traj_features_seg30`
    - det-th-15-5: `vidvrd_traj_roi_det_th-15-5.zip`, c.t. `data0/scene_graph_benchmark/output/VidVRD_traj_features_seg30_th-15-5`
- traj embds (256-d, and these are all filtered by th-15-5)
    - gt:  `vidvrd_traj_emb_det.zip`, c.t. `data0/ALPRO/extract_features_output/VidVRDtest_seg30_TrajFeatures256_gt`
    - det:  `vidvrd_traj_emb_det.zip`, c.t. `data0/ALPRO/extract_features_output/vidvrd_seg30_TrajFeatures256`

### Model weights
- TrajCls module
- RelationCls module

## VidOR


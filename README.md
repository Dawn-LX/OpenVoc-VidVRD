Official code for our ICLR2023 paper: "Compositional Prompt Tuning with Motion Cues for Open-Vocabulary Video Relation Detection"
[openreview link](https://openreview.net/pdf?id=mE91GkXYipg) 

# The code is still preparing .....

# Requirements

- Python == 3.7 or later, Pytorch == 1.7 or later
- transformers == 4.11.3 (new version might require some modifications for ALpro's code, but also works， refer to Line 872 in `Alpro_modeling/xbert.py`) 
- for other basic packages, just run the project and download whatever needed.

# data-release summarize

- `vidvrd_traj_box_det_th-15-5.zip`: this is used for TrajCls module only, it can be obtained by filter out trajs with length < 15 and area < 5, but we also provide this data to make sure.


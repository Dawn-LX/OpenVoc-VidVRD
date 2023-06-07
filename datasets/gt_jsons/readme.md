this dir should contains the following .json files

- VidVRDtest_gts.json
- VidORval_gts.json
- VidVRDtest_segment_gts.json
- VidORval_segment_gts.json

The first and second .json files can be obtained by following the previous [VidSGG-BIG](https://github.com/Dawn-LX/VidSGG-BIG#evaluation) repo

The third and fourth .json files are used for segment-level evaluation, which is for debug, and eventually the segment-level eval results is not used in our paper.  Neverthess, we can still build these `*_segment_gts.json` by this script [VidVRD-II-helper/prepare_gts_for_eval.py](https://github.com/Dawn-LX/OpenVoc-VidVRD/blob/master/VidVRD-II-helper/prepare_gts_for_eval.py)
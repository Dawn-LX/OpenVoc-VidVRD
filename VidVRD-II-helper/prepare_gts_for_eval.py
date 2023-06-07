from collections import defaultdict
from tqdm import tqdm
import json

from dataset import VidVRD, VidOR
from common.relation import VideoRelation
from common.misc import segment_video,get_segment_signature

def prepare_gts_for_vidvrd(save_path,segment_gt=True):
    dataset = VidVRD(
        "datasets/vidvrd-dataset", 
        'datasets/vidvrd-dataset/videos', 
        splits = ["train","test"],
        normalize_coords=False
    )
    indices = dataset.get_index(split="test")

    if segment_gt:
        ########### refer to func:`eval_relation_segments` in common/misc.py
        segment_level_gts = defaultdict(list)
        for vid in tqdm(indices):
            anno = dataset.get_anno(vid)
            segs = segment_video(0, anno['frame_count'])
            video_gts = dataset.get_relation_insts(vid)
            for fstart, fend in segs:
                vsig = get_segment_signature(vid, fstart, fend)
                for r_json in video_gts:
                    r = VideoRelation.from_json(r_json)
                    _r = r.get_relation_during(fstart, fend)
                    if _r is not None:
                        r_json = _r.serialize(allow_misalign=False)
                        if r_json is not None:
                            segment_level_gts[vsig].append(r_json)
        ########### 
        with open(save_path,'w') as f:
            json.dump(segment_level_gts,f)
    else:
        video_level_gts = dict()
        for vid in indices:
            video_level_gts[vid] = dataset.get_relation_insts(vid)
        with open(save_path,'w') as f:
            json.dump(video_level_gts,f)


def prepare_gts_for_vidor(save_path):
    # this func is copied from /home/gkf/project/VidSGG-BIG/VidVRD-helper/prepare_gts_for_eval.py
    
    dataset = VidOR(
        anno_rpath = "datasets/vidor-dataset/annotation", 
        video_rpath = 'datasets/vidor-dataset/val_videos', # videos for val
        # splits = ["training","validation"],
        splits = ["validation"],
        low_memory=False,
        normalize_coords=False,  ### NOTE here, default normalize_coords is True
    )
    indices = dataset.get_index(split="validation")


    video_level_gts = dict()
    for vid in indices:
        video_level_gts[vid] = dataset.get_relation_insts(vid)
    
    if save_path is not None:
        print("saving ...")
        with open(save_path,'w') as f:
            json.dump(video_level_gts,f)
        print("done.")

def prepare_gts_for_vidor_seg(save_path):
    dataset = VidOR(
        anno_rpath = "datasets/vidor-dataset/annotation", 
        video_rpath = 'datasets/vidor-dataset/val_videos', # videos for val
        # splits = ["training","validation"],
        splits = ["validation"],
        low_memory=False,
        normalize_coords=False
    )
    indices = dataset.get_index(split="validation")

    ########### refer to func:`eval_relation_segments` in common/misc.py
    segment_level_gts = defaultdict(list)
    for vid in tqdm(indices):
        anno = dataset.get_anno(vid)
        segs = segment_video(0, anno['frame_count'])
        video_gts = dataset.get_relation_insts(vid)
        for fstart, fend in segs:
            vsig = get_segment_signature(vid, fstart, fend)
            for r_json in video_gts:
                r = VideoRelation.from_json(r_json)
                _r = r.get_relation_during(fstart, fend)
                if _r is not None:
                    r_json = _r.serialize(allow_misalign=False)
                    if r_json is not None:
                        segment_level_gts[vsig].append(r_json)
    ########### 
    
    print("saving ...")
    with open(save_path,'w') as f:
        json.dump(segment_level_gts,f)
    print("done.")





if __name__ == "__main__":
    prepare_gts_for_vidvrd(
        save_path = "tmp/VidVRDtest_segment_gts.json",
        segment_gt = False
    )

    
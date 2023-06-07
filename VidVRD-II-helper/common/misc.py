import os

# Several module level utility functions

segment_length = 30
segment_stride = 15
rpath = '../'


def get_segment_signature(vid, fstart, fend):
    """
    Generating video clip signature string
    """
    return '{}-{:04d}-{:04d}'.format(vid, fstart, fend)


def get_feature_path(dataset, name, vid):
    """
    Path to save intermediate features
    """
    path = os.path.join(rpath, '{}-baseline-output'.format(dataset), 'features', name)
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, vid)
    if '.' not in vid:
        os.makedirs(path, exist_ok=True)
    return path


def get_model_path(model_id, dataset):
    """
    Path to save trained model
    """
    path = os.path.join(rpath, '{}-baseline-output'.format(dataset), 'models', model_id)
    os.makedirs(path, exist_ok=True)
    if not os.path.exists(os.path.join(path, 'weights')):
        os.makedirs(os.path.join(path, 'weights'))
    return path


def get_tensorflow_research_model_path():
    """
    Path to tensorflow_research_model for image object detection
    """
    return '../tensorflow-models/research/'


def get_object_detection_path(dataset, name, vid):
    """
    Path to save intermediate object detection results
    """
    path = os.path.join(rpath, '{}-baseline-output'.format(dataset), 'object_detection', name)
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, vid)
    if '.' not in vid:
        os.makedirs(path, exist_ok=True)
    return path


def segment_video(fstart, fend):
    """
    Given the duration [fstart, fend] of a video, segment the duration
    into many segments with overlapping
    """
    segs = [(i, i+segment_length) for i in range(fstart, fend-segment_length+1, segment_stride)]
    return segs


def segment_gt_relations(vid, relations, frame_count):
    gt_relation_segments = dict()
    segs = segment_video(0, frame_count)
    for fstart, fend in segs:
        vsig = get_segment_signature(vid, fstart, fend)

        segment_gts = []
        for r in relations:
            s = max(r['duration'][0], fstart)
            e = min(r['duration'][1], fend)
            if s<e:
                sub_trac = r['sub_traj'][s-r['duration'][0]: e-r['duration'][0]]
                obj_trac = r['obj_traj'][s-r['duration'][0]: e-r['duration'][0]]
                segment_gts.append({
                    "triplet": r['triplet'],
                    "subject_tid": r['subject_tid'],
                    "object_tid": r['object_tid'],
                    "duration": [s, e],
                    "sub_traj": sub_trac,
                    "obj_traj": obj_trac
                })
        if len(segment_gts) > 0:
            gt_relation_segments[vsig] = segment_gts

    return gt_relation_segments


def eval_relation_segments(dataset, indices, relation_segments, verbose=True):
    from tqdm import tqdm
    from collections import defaultdict
    from evaluation import eval_visual_relation, print_relation_scores
    from .relation import VideoRelation

    segment_gts = defaultdict(list)
    print('[info] segmenting GT relations')
    for vid in tqdm(indices):
        anno = dataset.get_anno(vid)
        segs = segment_video(0, anno['frame_count'])
        video_gts = dataset.get_relation_insts(vid)
        for fstart, fend in segs:
            vsig = get_segment_signature(vid, fstart, fend)
            for r_json in video_gts:
                # sub, pred, obj = r_json["triplet"]
                # r_json["triplet"] = [sub,"fg",obj]
                r = VideoRelation.from_json(r_json)
                _r = r.get_relation_during(fstart, fend)
                if _r is not None:
                    r_json = _r.serialize(allow_misalign=False)
                    if r_json is not None:
                        segment_gts[vsig].append(r_json)

    scores = dict()
    scores['overall'] = eval_visual_relation(segment_gts, relation_segments, allow_misalign=False, verbose=verbose)
    print_relation_scores(scores)

    return scores['overall']['detection mean AP']

## ------------ the following functions are added by gkf

def eval_relation_segments_OpenVoc(dataset, indices, relation_segments, dst_enti_cls,dst_pred_cls, verbose=True):
    from tqdm import tqdm
    from collections import defaultdict
    from evaluation import eval_visual_relation, print_relation_scores
    from .relation import VideoRelation


    segment_gts = defaultdict(list)
    print('[info] segmenting GT relations')
    for vid in tqdm(indices):
        anno = dataset.get_anno(vid)
        segs = segment_video(0, anno['frame_count'])
        video_gts = dataset.get_relation_insts(vid)
        for fstart, fend in segs:
            vsig = get_segment_signature(vid, fstart, fend)
            for r_json in video_gts:
                sub, pred, obj = r_json["triplet"]
                if not ((sub in dst_enti_cls) and (pred in dst_pred_cls) and (obj in dst_enti_cls)):
                    continue
                
                r = VideoRelation.from_json(r_json)
                _r = r.get_relation_during(fstart, fend)
                if _r is not None:
                    r_json = _r.serialize(allow_misalign=False)
                    if r_json is not None:
                        segment_gts[vsig].append(r_json)

    scores = dict()
    scores['overall'] = eval_visual_relation(segment_gts, relation_segments, allow_misalign=False, verbose=verbose)
    # print_relation_scores(scores)

    return scores
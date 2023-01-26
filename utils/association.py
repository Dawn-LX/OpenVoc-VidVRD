#####################
# this file is from VidVRD-II (https://dl.acm.org/doi/10.1145/3474085.3475263)
# combined based on `VidVRD-II/common/trajectory.py`, `VidVRD-II/common/relation.py`, and `VidVRD-II/common/association.py`
#####################
import copy
from collections import defaultdict

import numpy as np

from scipy.interpolate import interp1d

##### /home/gkf/project/VidVRD-II/common/trajectory.py

class Trajectory():
    """
    Object trajectory class that holds the bounding box trajectory and appearance feature (classeme)
    """
    def __init__(self, pstart, pend, rois, score=None, category=None, classeme=None, vsig=None, gt_trackid=-1):
        """
        bbox: [left, top, right, bottom]
        """
        pstart, pend = int(pstart), int(pend)  # satrt and end frame_id w.r.t video range
        assert len(rois)==pend-pstart
        self.pstart, self.pend = pstart, pend
        self.rois = [list(map(float, bbox)) for bbox in rois]  # fromat: xyxy
        self.score = score
        self.category = category
        self.classeme = classeme
        # video signature
        self.vsig = vsig
        self.gt_trackid = gt_trackid

    def __len__(self):
        return self.pend-self.pstart

    def __getitem__(self, p):
        """
        Return the bounding box at frame p
        """
        return self.rois[p-self.pstart]

    def __eq__(self, other):
        assert isinstance(other, self.__class__)
        if self.category != other.category:
            return False
        p_from = max(self.pstart, other.pstart)    
        p_to = min(self.pend, other.pend)
        if p_from < p_to:
            bboxes1 = []
            bboxes2 = []
            for p in range(p_from, p_to):
                roi1 = self[p]
                roi2 = other[p]
                bboxes1.append(roi1)
                bboxes2.append(roi2)
            bboxes1 = np.asarray([bboxes1], dtype=np.float32)
            bboxes2 = np.asarray([bboxes2], dtype=np.float32)
            iou = cubic_iou(bboxes1, bboxes2)
            assert iou.shape == (1, 1)
            return iou[0, 0] > 0.7
        else:
            return False

    def get_trajectory_during(self, pstart, pend):
        max_start = max(self.pstart, pstart)
        min_end = min(self.pend, pend)
        if max_start < min_end:
            rois = [self.rois[p-self.pstart] for p in range(max_start, min_end)]
            traj = Trajectory(max_start, min_end, rois, score=self.score, category=self.category, 
                    classeme=self.classeme, vsig=self.vsig, gt_trackid=self.gt_trackid)
            return traj
        else:
            return None
    
    def predict_trajectory_during(self, pstart, pend):
        assert pstart <= self.pstart and pend >= self.pend
        if len(self) == 1:
            predicted_rois = [self.rois[0]] * (pend-pstart)
        else:
            predicted_rois = []
            for c in range(4):
                y = np.asarray([bbox[c] for bbox in self.rois])
                f = interp1d([self.pstart, self.pend-1], [self.rois[0][c], self.rois[-1][c]],
                        kind='linear', fill_value='extrapolate', assume_sorted=True)
                before_y = np.clip(f(list(range(pstart, self.pstart))), 0, 1)
                after_y = np.clip(f(list(range(self.pend, pend))), 0, 1)
                predicted_rois.append(np.concatenate([before_y, y, after_y]))
            predicted_rois = np.asarray(predicted_rois).T
        
        traj = Trajectory(pstart, pend, predicted_rois, score=self.score, category=self.category, 
                classeme=self.classeme, vsig=self.vsig, gt_trackid=self.gt_trackid)
        return traj

    def temporal_intersection(self, other):
        min_start = min(self.pstart, other.pstart)
        min_end = min(self.pend, other.pend)
        max_start = max(self.pstart, other.pstart)
        max_end = max(self.pend, other.pend)
        return max((min_end-max_start)/(max_end-min_start), 0.)

    def predicted_cubic_intersection(self, other):
        min_start = min(self.pstart, other.pstart)
        max_end = max(self.pend, other.pend)
        traj1 = self.predict_trajectory_during(min_start, max_end)
        traj2 = other.predict_trajectory_during(min_start, max_end)
        bboxes1 = np.asarray([traj1.rois])
        bboxes2 = np.asarray([traj2.rois])
        iou = cubic_iou(bboxes1, bboxes2)[0, 0]
        return iou
    
    def cubic_intersection(self, other, temporal_tolerance=30):
        min_end = min(self.pend, other.pend)
        max_start = max(self.pstart, other.pstart)
        if max_start < min_end:
            traj1 = self.get_trajectory_during(max_start, min_end)
            traj2 = other.get_trajectory_during(max_start, min_end)
            bboxes1 = np.asarray([traj1.rois])
            bboxes2 = np.asarray([traj2.rois])
        elif max_start-min_end < temporal_tolerance:
            if self.pend <= other.pstart:
                bboxes1 = np.asarray([[self.rois[-1]]])
                bboxes2 = np.asarray([[other.rois[0]]])
            else:
                bboxes1 = np.asarray([[self.rois[0]]])
                bboxes2 = np.asarray([[other.rois[-1]]])
        else:
            return 0.
        iou = cubic_iou(bboxes1, bboxes2)[0, 0]
        return iou

    def cubic_enclose(self, other):
        min_end = min(self.pend, other.pend)
        max_start = max(self.pstart, other.pstart)
        if max_start < min_end:
            traj1 = self.get_trajectory_during(max_start, min_end)
            traj2 = other.get_trajectory_during(max_start, min_end)
            bboxes1 = np.asarray([traj1.rois]).transpose((1, 0, 2))
            bboxes2 = np.asarray([traj2.rois]).transpose((1, 0, 2))
            intersect_vol = _intersect(bboxes1, bboxes2)[0, 0]
            bboxes = np.asarray([other.rois]).transpose((1, 0, 2))
            self_vol = _union(bboxes, bboxes)[0, 0]*0.5
            return intersect_vol/max(self_vol, 1e-8)
        else:
            return 0.

    def join(self, other):
        assert self.category == other.category
        min_start = min(self.pstart, other.pstart)
        max_end = max(self.pend, other.pend)
        rois = []
        for p in range(min_start, max_end):
            if self.pstart<=p<self.pend and other.pstart<=p<other.pend:
                roi = [(c1+c2)/2 for c1, c2 in zip(self[p], other[p])]
            elif self.pstart<=p<self.pend:
                roi = list(self[p])
            elif other.pstart<=p<other.pend:
                roi = list(other[p])
            elif self.pend < other.pstart:
                roi = list(map(lambda c: np.interp(p, [self.pend-1, other.pstart], [self.rois[-1][c], other.rois[0][c]]), range(4)))
            elif self.pstart > other.pend:
                roi = list(map(lambda c: np.interp(p, [other.pend-1, self.pstart], [other.rois[-1][c], self.rois[0][c]]), range(4)))
            rois.append(roi)
        traj = Trajectory(min_start, max_end, rois, score=self.score, category=self.category, 
                classeme=self.classeme, vsig=self.vsig, gt_trackid=self.gt_trackid)
        return traj

    def serialize(self):
        traj = dict()
        traj['pstart'] = self.pstart
        traj['pend'] = self.pend
        traj['rois'] = [list(map(float, bbox)) for bbox in self.rois]
        if self.score:
            traj['score'] = float(self.score)
        if self.category:
            traj['category'] = self.category
        if self.classeme:
            traj['classeme'] = [float(x) for x in self.classeme]
        if self.vsig:
            traj['vsig'] = self.vsig
        traj['gt_trackid'] = self.gt_trackid
        return traj


def _intersect(bboxes1, bboxes2):
    """
    bboxes: t x n x 4
    """
    assert bboxes1.shape[0] == bboxes2.shape[0]
    t = bboxes1.shape[0]
    inters = np.zeros((bboxes1.shape[1], bboxes2.shape[1]), dtype = np.float32)
    _min = np.empty((bboxes1.shape[1], bboxes2.shape[1]), dtype = np.float32)
    _max = np.empty((bboxes1.shape[1], bboxes2.shape[1]), dtype = np.float32)
    w = np.empty((bboxes1.shape[1], bboxes2.shape[1]), dtype = np.float32)
    h = np.empty((bboxes1.shape[1], bboxes2.shape[1]), dtype = np.float32)
    for i in range(t):
        np.maximum.outer(bboxes1[i, :, 0], bboxes2[i, :, 0], out = _min)
        np.minimum.outer(bboxes1[i, :, 2], bboxes2[i, :, 2], out = _max)
        np.subtract(_max + 1, _min, out = w)
        w.clip(min = 0, out = w)
        np.maximum.outer(bboxes1[i, :, 1], bboxes2[i, :, 1], out = _min)
        np.minimum.outer(bboxes1[i, :, 3], bboxes2[i, :, 3], out = _max)
        np.subtract(_max + 1, _min, out = h)
        h.clip(min = 0, out = h)
        np.multiply(w, h, out = w)
        inters += w
    return inters


def _union(bboxes1, bboxes2):
    if id(bboxes1) == id(bboxes2):
        w = bboxes1[:, :, 2] - bboxes1[:, :, 0] + 1
        h = bboxes1[:, :, 3] - bboxes1[:, :, 1] + 1
        area = np.sum(w * h, axis = 0)
        unions = np.add.outer(area, area)
    else:
        w = bboxes1[:, :, 2] - bboxes1[:, :, 0] + 1
        h = bboxes1[:, :, 3] - bboxes1[:, :, 1] + 1
        area1 = np.sum(w * h, axis = 0)
        w = bboxes2[:, :, 2] - bboxes2[:, :, 0] + 1
        h = bboxes2[:, :, 3] - bboxes2[:, :, 1] + 1
        area2 = np.sum(w * h, axis = 0)
        unions = np.add.outer(area1, area2)
    return unions


def cubic_iou(bboxes1, bboxes2):
    # bboxes: n x t x 4 (left, top, right, bottom)
    if id(bboxes1) == id(bboxes2):
        bboxes1 = bboxes1.transpose((1, 0, 2))
        bboxes2 = bboxes1
    else:
        bboxes1 = bboxes1.transpose((1, 0, 2))
        bboxes2 = bboxes2.transpose((1, 0, 2))
    # compute cubic-IoU
    # bboxes: t x n x 4
    iou = _intersect(bboxes1, bboxes2)
    union = _union(bboxes1, bboxes2)
    np.subtract(union, iou, out = union)
    np.divide(iou, np.clip(union, 1e-8, None), out = iou)
    return iou


def traj_iou(trajs1, trajs2):
    """
    Compute the pairwise trajectory IoU in trajs1 and trajs2.
    Assumuing all trajectories in trajs1 and trajs2 start at same frame and
    end at same frame.
    """
    bboxes1 = np.asarray([traj.rois for traj in trajs1])
    if id(trajs1) == id(trajs2):
        bboxes2 = bboxes1
    else:
        bboxes2 = np.asarray([traj.rois for traj in trajs2])
    iou = cubic_iou(bboxes1, bboxes2)
    return iou


##### /home/gkf/project/VidVRD-II/common/relation.py

class VideoRelation():
    '''
    Represent video visual relation instances
    ----------
    Properties:
        sub - object class name for subject
        pred - predicate class name
        obj - object class name for object
        straj - the trajectory of subject
        otraj - the trajectory of object
        conf - confident score
        vsig - video clip signature
    '''

    @classmethod
    def from_json_original(cls, r_json):
        sub, pred, obj = r_json['triplet']
        if 'sub_duration' in r_json:
            sub_duration = r_json['sub_duration']
            obj_duration = r_json['obj_duration']
        else:
            sub_duration = r_json['duration']
            obj_duration = r_json['duration']
        straj = Trajectory(sub_duration[0], sub_duration[1], r_json['sub_traj'], category=sub)
        otraj = Trajectory(obj_duration[0], obj_duration[1], r_json['obj_traj'], category=obj)

        return cls(sub, pred, obj, straj, otraj, r_json.get('score', 0.))
    
    @classmethod
    def from_json(cls, r_json):
        '''
        modified by gkf
        I also modified self.__init__,  self.extend, self.serialize
        '''
        sub, pred, obj = r_json.pop("triplet")
        if 'sub_duration' in r_json:
            sub_duration = r_json.pop("sub_duration")
            obj_duration = r_json.pop("obj_duration")
        else:
            sub_duration = r_json.pop("duration")
            obj_duration = sub_duration
        straj = Trajectory(sub_duration[0], sub_duration[1], r_json.pop('sub_traj'), category=sub)
        otraj = Trajectory(obj_duration[0], obj_duration[1], r_json.pop('obj_traj'), category=obj)

        # added by gkf: 
        conf = r_json.pop("score")

        # other_infos = dict()
        # for key in r_json.keys():  # for other_info
        #     other_infos[key] = r_json.pop(key)
        
        other_infos = r_json
        return cls(sub, pred, obj, straj, otraj, conf, other_infos)

    def __init__(self, sub, pred, obj, straj, otraj, conf, other_infos=dict()):
        self.sub = sub
        self.pred = pred
        self.obj = obj
        self.confs_list = [conf]
        self.straj = straj
        self.otraj = otraj

        # added by gkf
        self.other_infos = dict()
        for name,info in other_infos.items():
            self.other_infos[name] = [info]
        
    
    def __repr__(self):
        return '<VideoRelation: {}({}-{}), {}, {}({}-{})>'.format(
                self.sub, self.straj.pstart, self.straj.pend,
                self.pred,
                self.obj, self.otraj.pstart, self.otraj.pend)

    def triplet(self):
        return (self.sub, self.pred, self.obj)
    
    def score(self):
        return sum(self.confs_list)
    
    def is_self_relation(self, iou_thr=0.9):
        return self.straj.cubic_intersection(self.otraj, temporal_tolerance=0) > iou_thr

    def overlap(self, other, iou_thr=0.5, temporal_tolerance=30):
        s_iou = self.straj.cubic_intersection(other.straj, temporal_tolerance=temporal_tolerance)
        if s_iou > iou_thr:
            o_iou = self.otraj.cubic_intersection(other.otraj, temporal_tolerance=temporal_tolerance)
            if o_iou > iou_thr:
                return True
        return False

    def enclose(self, other, iou_thr=0.5):
        s_iou = self.straj.cubic_enclose(other.straj)
        if s_iou > iou_thr:
            o_iou = self.otraj.cubic_enclose(other.otraj)
            if o_iou > iou_thr:
                return True
        return False

    def extend(self, other):
        self.straj = self.straj.join(other.straj)
        self.otraj = self.otraj.join(other.otraj)
        self.confs_list.append(other.score())

        # added by gkf
        for name in self.other_infos.keys():
            self.other_infos[name] = self.other_infos[name] + other.other_infos[name]

    def get_relation_during(self, pstart, pend):
        straj = self.straj.get_trajectory_during(pstart, pend)
        if straj is None:
            return None
        otraj = self.otraj.get_trajectory_during(pstart, pend)
        if otraj is None:
            return None
        return VideoRelation(self.sub, self.pred, self.obj, straj, otraj, self.score())

    def serialize(self, allow_misalign=False):
        rel = dict()
        rel['triplet'] = list(self.triplet())
        rel['score'] = float(self.score())
        if allow_misalign:
            rel['sub_duration'] = [self.straj.pstart, self.straj.pend]
            rel['obj_duration'] = [self.otraj.pstart, self.otraj.pend]
            rel['sub_traj'] = self.straj.serialize()['rois']
            rel['obj_traj'] = self.otraj.serialize()['rois']
        else:
            pstart = max(self.straj.pstart, self.otraj.pstart)
            pend = min(self.straj.pend, self.otraj.pend)
            if pend-pstart > 1:
                rel['duration'] = [pstart, pend]   # [start_fid,end_fid), end_fid is exclusive  # noted by gkf
                rel['sub_traj'] = self.straj.get_trajectory_during(pstart, pend).serialize()['rois']
                rel['obj_traj'] = self.otraj.get_trajectory_during(pstart, pend).serialize()['rois']
            else:
                # regarded as invalid video relations
                return None
        
        # added by gkf
        rel["score_list"] = self.confs_list
        
        # for name,info in self.other_infos.items():
        #     rel[name] = info
        # the above code is equal to
        rel.update(self.other_infos)
        
        return rel


##### /home/gkf/project/VidVRD-II/common/association.py

def parallel_association(vid, algorithm, relation_groups, param):
    try:
        return algorithm(relation_groups, **param)
    except Exception as e:
        print('[error] some problem found in processing {}. please stop manually to check'.format(vid))
        raise e


def greedy_graph_association(relation_groups, **param):
    video_segments = list(relation_groups.keys())
    video_segments.sort(key=lambda s: s[0]) # sort by fstart

    video_entity_list = []
    video_relation_list = []
    for fstart, fend in video_segments:
        video_relation_list.sort(key=lambda r: r.score(), reverse=True)

        relations = relation_groups[(fstart, fend)]
        sorted_relations = sorted(relations, key=lambda r: r['score'], reverse=True)
        sorted_relations = sorted_relations[:param['inference_topk']]
        
        cur_modify_rel_list = []
        for r_json in sorted_relations:
            this_r = VideoRelation.from_json(r_json)
            this_r.pstart = max(this_r.straj.pstart, this_r.otraj.pstart)
            this_r.pend = min(this_r.straj.pend, this_r.otraj.pend)

            for eid, e in enumerate(video_entity_list):
                if e.category == this_r.sub and e.cubic_intersection(this_r.straj) > param['association_linkage_threshold']:
                    video_entity_list[eid] = e.join(this_r.straj)
                    this_r.straj = eid
                    break
            else:
                this_r.straj.category = this_r.sub
                video_entity_list.append(this_r.straj)
                this_r.straj = len(video_entity_list)-1

            for eid, e in enumerate(video_entity_list):
                if e.category == this_r.obj and e.cubic_intersection(this_r.otraj) > param['association_linkage_threshold']:
                    video_entity_list[eid] = e.join(this_r.otraj)
                    this_r.otraj = eid
                    break
            else:
                this_r.otraj.category = this_r.obj
                video_entity_list.append(this_r.otraj)
                this_r.otraj = len(video_entity_list)-1

            if this_r.pstart < this_r.pend and this_r.straj != this_r.otraj:
                for last_r in video_relation_list:
                    if last_r.triplet() == this_r.triplet() and last_r.straj == this_r.straj and last_r.otraj == this_r.otraj:
                        if last_r.pstart < this_r.pstart:
                            min_start, max_start = last_r.pstart, this_r.pstart
                        else:
                            min_start, max_start = this_r.pstart, last_r.pstart
                        if last_r.pend < this_r.pend:
                            min_end, max_end = last_r.pend, this_r.pend
                        else:
                            min_end, max_end = this_r.pend, last_r.pend
                        if max_start <= min_end:
                            last_r.pstart = min_start
                            last_r.pend = max_end
                            last_r.confs_list.append(this_r.score())
                            break
                else:
                    video_relation_list.append(this_r)
    
    entities = []
    trajectories = defaultdict(list)
    for eid, e in enumerate(video_entity_list):
        entities.append({
            'tid': eid,
            'category': e.category
        })
        for i, bbox in enumerate(e.rois):
            trajectories[e.pstart+i].append({
                'tid': eid,
                'bbox': {
                    'xmin': float(bbox[0]),
                    'ymin': float(bbox[1]),
                    'xmax': float(bbox[2]),
                    'ymax': float(bbox[3])
                }
            })
    
    video_relation_list.sort(key=lambda r: r.score(), reverse=True)
    relation_instances = []
    for r in video_relation_list[:param['association_topk']]:
        relation_instances.append({
            'subject_tid': r.straj,
            'object_tid': r.otraj,
            'predicate': r.pred,
            'score': r.score(),
            'begin_fid': r.pstart,
            'end_fid': r.pend
        })

    graph = dict()
    graph['subject/objects'] = entities
    graph['trajectories'] = [trajectories[fid] for fid in range(video_segments[-1][1])]
    graph['relation_instances'] = relation_instances

    return graph


def greedy_relation_association(relation_groups, **param):
    video_segments = list(relation_groups.keys())
    video_segments.sort(key=lambda s: s[0]) # sort by fstart

    video_relation_list = []
    last_modify_rel_list = []
    for fstart, fend in video_segments:
        last_modify_rel_list.sort(key=lambda r: r.score(), reverse=True)

        relations = relation_groups[(fstart, fend)]
        sorted_relations = sorted(relations, key=lambda r: r['score'], reverse=True)
        sorted_relations = sorted_relations[:param['inference_topk']]
        
        cur_modify_rel_list = []
        for r_json in sorted_relations:
            this_r = VideoRelation.from_json(r_json)
            for last_r in last_modify_rel_list:
                if last_r.triplet() == this_r.triplet() and last_r.overlap(this_r, iou_thr=param['association_linkage_threshold']):
                    last_r.extend(this_r)  # here, score merge is sum
                    cur_modify_rel_list.append(last_r)
                    break
            else:
                video_relation_list.append(this_r)
                cur_modify_rel_list.append(this_r)
                continue
            last_modify_rel_list.remove(last_r)
        last_modify_rel_list = cur_modify_rel_list
    
    results = []
    for r in video_relation_list:
        r_json = r.serialize(allow_misalign=False)
        if r_json is not None:
            results.append((r, r_json))
    results.sort(key=lambda r: r[1]['score'], reverse=True)
    results = results[:param['association_topk']]

    if param['association_nms'] < 1:
        order = list(range(len(results)))
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(results[i])
            this_r = results[i][0]
            new_order = []
            for j in order[1:]:
                other = results[j][0]
                if this_r.triplet() == other.triplet() and this_r.enclose(other, iou_thr=param['association_nms']):
                    continue
                new_order.append(j)
            order = new_order
        results = keep

    return [r_json for _, r_json in results]


def nms_relation_association(relation_groups, score_metric='avg', **param):
    video_segments = list(relation_groups.keys())
    video_segments.sort(key=lambda s: s[0]) # sort by fstart

    relation_segments = []
    scores = []
    for fstart, fend in video_segments:
        r_jsons = relation_groups[(fstart, fend)]
        sorted_r_jsons = sorted(r_jsons, key=lambda r: r['score'], reverse=True)
        relations = [VideoRelation.from_json(r_json) for r_json in sorted_r_jsons[:param['inference_topk']]]
        relation_segments.append(relations)
        scores.append([r.score() for r in relations])

    graph = build_linkage_graph(relation_segments, param['association_linkage_threshold'])
    video_relation_list = []
    for _ in range(param['association_topk']): 
        seg_start_index, best_association, best_score = find_best_association(graph, scores)
        if best_score <= 0:
            break 
        # merge relation segments and rescore
        best_relation = copy.copy(relation_segments[seg_start_index][best_association[0]])
        for i in range(1, len(best_association)):
            r = relation_segments[seg_start_index+i][best_association[i]]
            best_relation.extend(r.straj, r.otraj, r.score())
        video_relation_list.append(best_relation)
        # supress overlapped relation segments
        graph, scores = suppression(best_relation, best_association, seg_start_index, relation_segments,
                graph, scores, suppress_threshold=param['association_nms'])
    
    results = []
    for r in video_relation_list:
        r_json = r.serialize(allow_misalign=False)
        if r_json is not None:
            results.append(r_json)

    return results


def build_linkage_graph(relation_segments, linkage_threshold=0.5):
    graph = []
    for i in range(len(relation_segments)-1):
        adjacency_matrix = []
        for r_i in relation_segments[i]:
            edges = []
            for idx, r_i1 in enumerate(relation_segments[i+1]):
                if r_i.triplet()==r_i1.triplet() and r_i.both_overlap(r_i1.straj, r_i1.otraj, iou_thr=linkage_threshold):
                    edges.append(idx)
            adjacency_matrix.append(edges)
        graph.append(adjacency_matrix)

    return graph


def suppression(relation, association_to_delete, seg_start_index, relation_segments, graph, scores, suppress_threshold=0.3):
    for i in range(seg_start_index, seg_start_index+len(association_to_delete)):
        deletes = []
        for idx, other in enumerate(relation_segments[i]):
            if scores[i][idx]>0 and relation.triplet()==other.triplet() and\
                    relation.both_overlap(other.straj, other.otraj, iou_thr=suppress_threshold, temporal_tolerance=0):
                deletes.append(idx)

        for delete_idx in deletes:
            scores[i][delete_idx] = 0.
        if i < len(graph): 
            for delete_idx in deletes:
                graph[i][delete_idx] = []
        if i > 0 or seg_start_index > 0:
            # remove connections to current sequence node from previous frame nodes
            for prior_box in graph[i-1]: 
                for delete_idx in deletes:
                    if delete_idx in prior_box:
                        prior_box.remove(delete_idx)

    return graph, scores


def find_best_association(graph, scores):
    ''' Given graph of all linked boxes, find the best sequence in the graph. The best sequence 
    is defined as the sequence with the maximum score across an arbitrary number of frames.
    We build the sequences back to front from the last frame to easily capture start of new sequences/
    Condition to start of new sequence: 
        if there are no edges from boxes in frame t-1, then the box in frame t must be the start of a new sequence.
        This assumption is valid since all scores are positive so we can always improve a sequence by increasing its length. 
        Therefore if there are links to a box from previous frames, we can always build a better path by extending it s.t. 
        the box cannot be the start of a new best sequence. 
    Args
        graph             : list of shape (num_frames - 1, num_boxes, k) returned from build_box_sequences that contains box sequences 
        scores                : Tensor of shape (num_frames, num_boxes) containing the label for the corresponding box. 
    Returns 
        None
    '''
    # list of tuples storing (score up to current frame, path up to current frame)
    # we dynamically build up best paths through graph starting from the end frame
    # s.t we can determine the beginning of sequences i.e. if there are no links 
    # to a box from previous frames, then it is a candidate for starting a sequence 
    max_scores_paths = [] 

    # list of all independent sequences where a given row corresponds to starting frame
    sequence_roots = []

    # starting from the last frame, build base paths i.e paths consisting of a single node 
    max_scores_paths.append([(score, [idx]) for idx, score in enumerate(scores[-1])])

    for reverse_idx, frame_edges in enumerate(graph[::-1]): # list of edges between neigboring frames i.e frame dimension 
        max_paths_f = []
        used_in_sequence = np.zeros(len(max_scores_paths[-1]), int)
        frame_idx = len(graph) - reverse_idx - 1
        for box_idx, box_edges in enumerate(frame_edges): # list of edges for each box in frame i.e. box dimension
            if not box_edges: # no edges for current box so consider it a max path consisting of a single node 
                max_paths_f.append((scores[frame_idx][box_idx], [box_idx]))
            else: # extend previous max paths 
                # here we use box_edges list to index used_in_sequence list and mark boxes in corresponding frame t+1 
                # as part of a sequence since we have links to them and can always make a better max path by making it longer (no negative scores)
                used_in_sequence[box_edges] = 1
                prev_idx = np.argmax([max_scores_paths[-1][bidx][0] for bidx in box_edges])
                score_so_far = max_scores_paths[-1][box_edges[prev_idx]][0]
                path_so_far = copy.copy(max_scores_paths[-1][box_edges[prev_idx]][1])
                path_so_far.append(box_idx)
                max_paths_f.append((scores[frame_idx][box_idx] + score_so_far, path_so_far))
        
        # create new sequence roots for boxes in frame at frame_idx + 1 that did not have links from boxes in frame_idx
        new_sequence_roots = [max_scores_paths[-1][idx] for idx, flag in enumerate(used_in_sequence) if flag == 0]

        sequence_roots.append(new_sequence_roots) 
        max_scores_paths.append(max_paths_f)
    
    # add sequences starting in begining frame as roots 
    sequence_roots.append(max_scores_paths[-1])

    # reverse sequence roots since built sequences from back to front 
    sequence_roots = sequence_roots[::-1]

    # iterate sequence roots to find sequence with max score 
    best_score = 0 
    best_association = [] 
    seg_start_index = 0
    for index, associations in enumerate(sequence_roots):
        if not associations: continue 
        max_index = np.argmax([sequence[0] for sequence in associations])
        if associations[max_index][0] > best_score:
            best_score = associations[max_index][0]
            best_association = associations[max_index][1][::-1] # reverse path 
            seg_start_index = index
    return seg_start_index, best_association, best_score

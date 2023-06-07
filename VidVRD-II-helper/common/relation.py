from .trajectory import Trajectory


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
    def from_json(cls, r_json):
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

    def __init__(self, sub, pred, obj, straj, otraj, conf):
        self.sub = sub
        self.pred = pred
        self.obj = obj
        self.confs_list = [conf]
        self.straj = straj
        self.otraj = otraj
    
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
        
        return rel

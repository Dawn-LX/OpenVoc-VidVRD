import os
import json

import numpy as np
from scipy.interpolate import interp1d


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

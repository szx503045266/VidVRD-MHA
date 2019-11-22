import itertools
from copy import deepcopy
from collections import deque

from dlib import drectangle

from .trajectory import traj_iou


def _merge_trajs(traj_1, traj_2):
    try:
        assert traj_1.pend > traj_2.pstart and traj_1.pstart < traj_2.pend
    except AssertionError:
        print('{}-{} {}-{}'.format(traj_1.pstart, traj_1.pend, traj_2.pstart, traj_2.pend))
    overlap_length = max(traj_1.pend - traj_2.pstart, 0)
    for i in range(overlap_length):
        roi_1 = traj_1.rois[traj_1.length() - overlap_length + i]
        roi_2 = traj_2.rois[i]
        left = (roi_1.left() + roi_2.left()) / 2
        top = (roi_1.top() + roi_2.top()) / 2
        right = (roi_1.right() + roi_2.right()) / 2
        bottom = (roi_1.bottom() + roi_2.bottom()) / 2
        traj_1.rois[traj_1.length() - overlap_length + i] = drectangle(left, top, right, bottom)
    for i in range(overlap_length, traj_2.length()):
        traj_1.predict(traj_2.rois[i])
    return traj_1


def _traj_iou_over_common_frames(traj_1, traj_2):
    if traj_1.pend <= traj_2.pstart or traj_2.pend <= traj_1.pstart: # no overlap
        return 0
    if traj_1.pstart <= traj_2.pstart:
        t1 = deepcopy(traj_1)
        t2 = deepcopy(traj_2)
    else:
        t1 = deepcopy(traj_2)
        t2 = deepcopy(traj_1)
    overlap_length = t1.pend - t2.pstart
    t1.rois = deque(itertools.islice(t1.rois, t2.pstart-t1.pstart, t1.pend-t1.pstart))
    t2.rois = deque(itertools.islice(t2.rois, 0, t1.pend-t2.pstart))
    iou = traj_iou([t1], [t2])
    return iou[0,0]


class VideoRelation(object):
    '''
    Represent video visual relation instances
    ----------
    Properties:
        vid - video name
        sub - object class name for subject
        pred - predicate class name
        obj - object class name for object
        straj - the trajectory of subject
        otraj - the trajectory of object
        conf - confident score
    '''
    def __init__(self, sub, pred, obj, straj, otraj, conf):
        self.sub = sub
        self.pred = pred
        self.obj = obj
        self.straj = straj
        self.otraj = otraj
        self.confs_list = [conf]
        assert straj.pstart == otraj.pstart
        assert straj.pend == otraj.pend
        self.fstart = straj.pstart
        self.fend = straj.pend
    
    def __repr__(self):
        return '<VideoRelation {} {}-{}-{}>'.format(
                self.straj.vsig, self.sub, self.pred, self.obj)

    def triplet(self):
        return (self.sub, self.pred, self.obj)
    
    def score(self):
        return sum(self.confs_list) / len(self.confs_list)
    
    def both_overlap(self, straj, otraj, iou_thr=0.5):
        s_iou = _traj_iou_over_common_frames(self.straj, straj)
        o_iou = _traj_iou_over_common_frames(self.otraj, otraj)
        return s_iou >= iou_thr and o_iou >= iou_thr

    def extend(self, straj, otraj, conf):
        self.straj = _merge_trajs(self.straj, straj)
        self.otraj = _merge_trajs(self.otraj, otraj)
        self.confs_list.append(conf)
        self.fstart = self.straj.pstart
        self.fend = self.otraj.pend

    def serialize(self):
        obj = dict()
        obj['triplet'] = list(self.triplet())
        obj['score'] = float(self.score())
        obj['duration'] = [
            int(self.fstart),
            int(self.fend)
        ]
        obj['sub_traj'] = self.straj.serialize()['rois']
        obj['obj_traj'] = self.otraj.serialize()['rois']
        return obj
from collections import defaultdict

import numpy as np

from baseline import *
from .trajectory import Trajectory
from .relation import VideoRelation


def greedy_relational_association(short_term_relations, truncate_per_segment=100):
    # group short-term relations by their staring frames
    pstart_relations = defaultdict(list)
    for r in short_term_relations:
        pstart_relations[r['duration'][0]].append(r)

    video_relation_list = []
    last_modify_rel_list = []
    for pstart in sorted(pstart_relations.keys()):
        last_modify_rel_list.sort(key=lambda r: r.score(), reverse=True)
        sorted_relations = sorted(pstart_relations[pstart], key=lambda r: r['score'], reverse=True)
        sorted_relations = sorted_relations[:truncate_per_segment]
        
        cur_modify_rel_list = []
        for rel in sorted_relations:
            conf_score = rel['score']
            sub, pred, obj = rel['triplet']
            _, pend = rel['duration']
            straj = Trajectory(pstart, pend, rel['sub_traj'])
            otraj = Trajectory(pstart, pend, rel['obj_traj'])
            
            for r in last_modify_rel_list:
                if r.triplet() == tuple(rel['triplet']) and r.both_overlap(straj, otraj, iou_thr=0.5):
                    # merge
                    r.extend(straj, otraj, conf_score)
                    last_modify_rel_list.remove(r)
                    cur_modify_rel_list.append(r)
                    break
            else:
                r = VideoRelation(sub, pred, obj, straj, otraj, conf_score)
                video_relation_list.append(r)
                cur_modify_rel_list.append(r)

        last_modify_rel_list = cur_modify_rel_list
    
    return [r.serialize() for r in video_relation_list]

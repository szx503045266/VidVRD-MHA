from baseline import *
from baseline.association import *
from baseline.trajectory import *

def dummy_association(dataset, short_term_relations, max_traj_num_in_clip=100):
    vrelation_list = []

    for index, prediction in short_term_relations:
        # load prediction data
        vid, fstart, fend = index
        pred_list, iou, trackid = prediction
        pred_list = sorted(pred_list, key=lambda x: x[0], reverse=True)
        if len(pred_list) > max_traj_num_in_clip:
            pred_list = pred_list[0:max_traj_num_in_clip]
        # load predict trajectory data
        trajs = object_trajectory_proposal(dataset, vid, fstart, fend)
        for traj in trajs:
            traj.pstart = fstart
            traj.pend = fend
            traj.vsig = get_segment_signature(vid, fstart, fend)
        
        for pred_idx, pred in enumerate(pred_list):
            conf_score = pred[0]
            s_cid, pid, o_cid = pred[1]
            s_tididx, o_tididx = pred[2]
            s_traj = trajs[s_tididx]
            o_traj = trajs[o_tididx]

            vrelation = dict()
            vrelation['triplet'] = [
                dataset.get_object_name(s_cid),
                dataset.get_predicate_name(pid),
                dataset.get_object_name(o_cid)
            ]
            vrelation['score'] = float(conf_score)
            vrelation['duration'] = [
                int(fstart),
                int(fend)
            ]
            vrelation['sub_traj'] = s_traj.serialize()['rois']
            vrelation['obj_traj'] = o_traj.serialize()['rois']
            
            vrelation_list.append(vrelation)
            #print(vrelation['duration'],vrelation['triplet'],vrelation['score'])

    #print(len(vrelation_list))
    return vrelation_list
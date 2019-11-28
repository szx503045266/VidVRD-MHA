import os
import json
import pickle
import argparse
from collections import defaultdict

from tqdm import tqdm

from dataset import VidVRD
from baseline_new import segment_video, get_model_path, get_segment_signature
from baseline_new import trajectory, feature, model, association
from evaluation import eval_video_object, eval_visual_relation

from mht_simple_new import *

def load_relation_feature():
    """
    Test loading precomputed relation features
    """
    dataset = VidVRD('/home/szx/vidvrd-dataset', '/home/szx/vidvrd-dataset/videos', ['train', 'test'])
    extractor = feature.FeatureExtractor(dataset, prefetch_count=0)

    video_indices = dataset.get_index(split='train')
    for vid in video_indices:
        durations = set(rel_inst['duration'] for rel_inst in dataset.get_relation_insts(vid, no_traj=True))
        for duration in durations:
            segs = segment_video(*duration)
            for fstart, fend in segs:
                extractor.extract_feature(dataset, vid, fstart, fend, verbose=True)

    video_indices = dataset.get_index(split='test')
    for vid in video_indices:
        anno = dataset.get_anno(vid)
        segs = segment_video(0, anno['frame_count'])
        for fstart, fend in segs:
            extractor.extract_feature(dataset, vid, fstart, fend, verbose=True)


def _convert_tracklet_for_evaluation(dataset, vsig, tracklets):
    converted = []
    for t in tracklets:
        assert t.vsig==vsig
        c = {
            'category': dataset.get_object_name(t.category),
            'score': t.score,
            'trajectory': dict()
        }
        for i, r in enumerate(t.rois):
            c['trajectory']['{}'.format(t.pstart+i)] = [
                round(r.left()), round(r.top()), round(r.right()), round(r.bottom())
            ]
        converted.append(c)
    return converted


def eval_object_tracklet_proposal():
    """
    Test loading and evaluate precomputed object tracklet proposals
    """
    dataset = VidVRD('/home/szx/vidvrd-dataset', '/home/szx/vidvrd-dataset/videos', ['train', 'test'])
    groundtruth = dict()
    prediction = dict()

    segment_indices = []
    video_indices = dataset.get_index(split='train')
    for vid in video_indices:
        durations = set(rel_inst['duration'] for rel_inst in dataset.get_relation_insts(vid, no_traj=True))
        for duration in durations:
            segs = segment_video(*duration)
            for fstart, fend in segs:
                segment_indices.append((vid, fstart, fend))
    video_indices = dataset.get_index(split='test')
    for vid in video_indices:
        anno = dataset.get_anno(vid)
        segs = segment_video(0, anno['frame_count'])
        for fstart, fend in segs:
            segment_indices.append((vid, fstart, fend))

    print('loading precomputed tracklets...')
    for vid, fstart, fend in tqdm(segment_indices):
        vsig = get_segment_signature(vid, fstart, fend)
        # get predicted tracklets in short-term
        tracklets = trajectory.object_trajectory_proposal(dataset, vid, fstart, fend, gt=False)
        prediction[vsig] = _convert_tracklet_for_evaluation(dataset, vsig, tracklets)
        # get ground truth tracklets in short-term
        tracklets = trajectory.object_trajectory_proposal(dataset, vid, fstart, fend, gt=True)
        groundtruth[vsig] = _convert_tracklet_for_evaluation(dataset, vsig, tracklets)

    mean_ap, ap_class = eval_video_object(groundtruth, prediction, thresh_t=0.5)


def eval_short_term_relation():
    """
    Evaluate short-term relation prediction
    """
    dataset = VidVRD('/home/szx/vidvrd-dataset', '/home/szx/vidvrd-dataset/videos', ['train', 'test'])
    with open(os.path.join(get_model_path(), 'baseline_setting.json'), 'r') as fin:
        param = json.load(fin)
    
    res_path = os.path.join(get_model_path(), 'short_term_relations.json')
    if os.path.exists(res_path):
        with open(res_path, 'r') as fin:
            short_term_relations = json.load(fin)['results']
    else:
        short_term_relations = model.predict(dataset, param)
        with open(res_path, 'w') as fout:
            json.dump({'results': short_term_relations}, fout)

    short_term_gt = dict()
    short_term_pred = dict()
    video_indices = dataset.get_index(split='test')
    for vid in video_indices:
        anno = dataset.get_anno(vid)
        segs = segment_video(0, anno['frame_count'])
        video_gts = dataset.get_relation_insts(vid)
        video_preds = short_term_relations[vid]
        for fstart, fend in segs:
            vsig = get_segment_signature(vid, fstart, fend)

            segment_gts = []
            for r in video_gts:
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
            short_term_gt[vsig] = segment_gts

            segment_preds = []
            for r in video_preds:
                if fstart<=r['duration'][0] and r['duration'][1]<=fend:
                    s = max(r['duration'][0], fstart)
                    e = min(r['duration'][1], fend)
                    sub_trac = r['sub_traj'][s-r['duration'][0]: e-r['duration'][0]]
                    obj_trac = r['obj_traj'][s-r['duration'][0]: e-r['duration'][0]]
                    segment_preds.append({
                        "triplet": r['triplet'],
                        "score": r['score'],
                        "duration": [s, e],
                        "sub_traj": sub_trac,
                        "obj_traj": obj_trac
                    })
            short_term_pred[vsig] = segment_preds

    mean_ap, rec_at_n, mprec_at_n = eval_visual_relation(short_term_gt, short_term_pred)


def train():
    dataset = VidVRD('/home/szx/vidvrd-dataset', '/home/szx/vidvrd-dataset/videos', ['train', 'test'])

    param = dict()
    param['model_name'] = 'baseline'
    param['rng_seed'] = 1701
    param['max_sampling_in_batch'] = 32
    param['batch_size'] = 64
    param['learning_rate'] = 0.001
    param['weight_decay'] = 0.0
    param['max_iter'] = 5000
    param['display_freq'] = 1
    param['save_freq'] = 5000
    param['epsilon'] = 1e-8
    param['pair_topk'] = 20
    param['seg_topk'] = 100
    param['video_topk'] = 200
    print(param)

    model.train(dataset, param)


def detect():
    dataset = VidVRD('/home/szx/vidvrd-dataset', '/home/szx/vidvrd-dataset/videos', ['train', 'test'])
    with open(os.path.join(get_model_path(), 'baseline_setting.json'), 'r') as fin:
        param = json.load(fin)
    '''
    short_term_relations = model.predict(dataset, param)
    with open(os.path.join(get_model_path(), 'short_term_relations.pkl'), 'wb') as fout:
        pickle.dump(short_term_relations, fout)
    '''
    with open(os.path.join(get_model_path(), 'short_term_relations.pkl'), 'rb') as fin:
        short_term_relations = pickle.load(fin)
    
    # video-level visual relation detection by relational association
    print('greedy relational association ...')
    video_relations = dict()
    for vid in tqdm(short_term_relations.keys()):
        res = association.greedy_relational_association(short_term_relations[vid], param['seg_topk'])
        res = sorted(res, key=lambda r: r['score'], reverse=True)[:param['video_topk']]
        video_relations[vid] = res
    # save detection result
    with open(os.path.join(get_model_path(), 'baseline_relation_prediction_new.json'), 'w') as fout:
        output = {
            'version': 'VERSION 1.0',
            'results': video_relations
        }
        json.dump(output, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VidVRD baseline')
    parser.add_argument('--load_feature', action="store_true", default=False, help='Test loading precomputed features')
    parser.add_argument('--eval_tracklet', action="store_true", default=False, help='Evaluate precomputed tracklet proposals')
    parser.add_argument('--eval_short_relation', action="store_true", default=False, help='Evaluate short-term relation prediction')
    parser.add_argument('--train', action="store_true", default=False, help='Train model')
    parser.add_argument('--detect', action="store_true", default=False, help='Detect video visual relation')
    args = parser.parse_args()

    if args.load_feature:
        load_relation_feature()
    elif args.eval_tracklet:
        eval_object_tracklet_proposal()
    elif args.eval_short_relation:
        eval_short_term_relation()
    elif args.train:
        train()
    elif args.detect:
        detect()
    else:
        parser.print_help()

import os
import json
import argparse
import gensim
import pickle as pkl
import numpy as np
import multiprocessing
from collections import defaultdict

from tqdm import tqdm

from dataset import VidVRD
from baseline import segment_video, get_model_path
from baseline import trajectory, feature, model, association

from mht_simple_all_conn import *
from bottomline import *

def load_object_trajectory_proposal():
    """
    Test loading precomputed object trajectory proposals
    """
    dataset = VidVRD('/home/szx/vidvrd-dataset', '/home/szx/vidvrd-dataset/videos', ['train', 'test'])

    video_indices = dataset.get_index(split='train')
    for vid in video_indices:
        durations = set(rel_inst['duration'] for rel_inst in dataset.get_relation_insts(vid, no_traj=True))
        for duration in durations:
            segs = segment_video(*duration)
            for fstart, fend in segs:
                trajs = trajectory.object_trajectory_proposal(dataset, vid, fstart, fend, gt=False, verbose=True)
                trajs = trajectory.object_trajectory_proposal(dataset, vid, fstart, fend, gt=True, verbose=True)

    video_indices = dataset.get_index(split='test')
    for vid in video_indices:
        anno = dataset.get_anno(vid)
        segs = segment_video(0, anno['frame_count'])
        for fstart, fend in segs:
            trajs = trajectory.object_trajectory_proposal(dataset, vid, fstart, fend, gt=False, verbose=True)
            trajs = trajectory.object_trajectory_proposal(dataset, vid, fstart, fend, gt=True, verbose=True)


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
    param['seg_topk'] = 200
    print(param)

    model.train(dataset, param)


def detect():
    dataset = VidVRD('/home/szx/vidvrd-dataset', '/home/szx/vidvrd-dataset/videos', ['train', 'test'])
    '''
    with open(os.path.join(get_model_path(), 'baseline_setting.json'), 'r') as fin:
        param = json.load(fin)
    short_term_relations = model.predict(dataset, param)
    # group short term relations by video
    video_st_relations = defaultdict(list)
    for index, st_rel in short_term_relations.items():
        vid = index[0]
        video_st_relations[vid].append((index, st_rel))
    #print(video_st_relations['ILSVRC2015_train_00914000'][0])
    
    with open(os.path.join(get_model_path(), 'video_st_relations_sotop2.pkl'), 'wb') as fout:
        pkl.dump(video_st_relations, fout)
    
    '''
    
    with open(os.path.join(get_model_path(), 'video_st_relations_sotop2.pkl'), 'rb') as fin:
        video_st_relations = pkl.load(fin)
    '''
    with open('/home/szx/Files/VVRD/vidvrd-baseline-output/datasetGCN/test_result/video_st_relations.pkl', 'rb') as fin:
        video_st_relations = pkl.load(fin)
    '''
    #Word2vec
    '''
    print('loading word vectors...')
    word_vectors = gensim.models.KeyedVectors.load_word2vec_format('word2vec/GoogleNews-vectors-negative300.bin.gz', binary=True)
    print('Word vectors loaded.')
    '''
    
    # video-level visual relation detection by relational association
    print('mht relational association ...')
    
    vid_list, video_st_relations_list = list(zip(*list(video_st_relations.items())))
    res_list = []
    pool = multiprocessing.Pool(processes=12)
    for re in video_st_relations_list:
        res_list.append(pool.apply_async(mht_association, (dataset, re, [])))
    pool.close()
    pool.join()

    video_relations = dict()
    for i, vid in enumerate(vid_list):
        video_relations[vid] = res_list[i].get()
    
    '''
    video_relations = dict()
    
    for vid in tqdm(video_st_relations.keys()):
        video_relations[vid] = mht_association(dataset, video_st_relations[vid])
    #video_relations['ILSVRC2015_train_00914000'] = mht_association(dataset, video_st_relations['ILSVRC2015_train_00914000'])
    '''
    
    # save detection result
    with open(os.path.join(get_model_path(), 'mht_relation_prediction.json'), 'w') as fout:
        output = {
            'version': 'VERSION 1.0',
            'results': video_relations
        }
        json.dump(output, fout)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VidVRD baseline')
    parser.add_argument('--load_feature', action="store_true", default=False, help='Test loading precomputed features')
    parser.add_argument('--train', action="store_true", default=False, help='Train model')
    parser.add_argument('--detect', action="store_true", default=False, help='Detect video visual relation')
    args = parser.parse_args()

    if args.load_feature or args.train or args.detect:
        if args.load_feature:
            load_object_trajectory_proposal()
            load_relation_feature()
        if args.train:
            train()
        if args.detect:
            detect()
    else:
        parser.print_help()

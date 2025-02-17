import os
import sys
#sys.path.append("/home/szx/Files/VVRD/VidVRD-helper")

import pickle
import argparse
import itertools
import time
from collections import deque, defaultdict, Counter
from copy import deepcopy
#from sklearn.metrics.pairwise import cosine_similarity

from dlib import drectangle
import numpy as np

#import gensim

from baseline import *
from baseline.association import *
from baseline.association import _traj_iou, _merge_trajs
from baseline.trajectory import *

#from util.trajectory import *
#from association.videorelation import _merge_trajs, _traj_iou

class Node(object):
    
    def __init__(self, id, segment, parent_id, child_id, s, p, o, s_score, p_score, o_score, s_traj, o_traj, rel_score):
        # child_id is a list
        self.id = id
        self.segment = segment
        self.parent_id = parent_id
        self.child_id = child_id
        self.s = s
        self.p = p
        self.o = o
        self.s_score = s_score
        self.p_score = p_score
        self.o_score = o_score
        self.s_traj = s_traj
        self.o_traj = o_traj
        self.rel_score = rel_score

class Tree(object):

    def __init__(self, segment, id, s, p, o, s_score, p_score, o_score, s_traj, o_traj, conf_score):
        self.nodes = dict()
        self.nodes[id] = Node(id, segment, -1, [], s, p, o, s_score, p_score, o_score, s_traj, o_traj, (conf_score, conf_score/10))
        self.node_num = 1
        self.leaflist = [id]

    def add_segment(self, vrelation_list, gt, segment, start_id, num, sorted_pred_list, trajs, word_vectors):
        '''The input are rel_feat, s_traj, o_traj of all tracklets in this segment.
        Return the index of the node not linked'''
        success = []
        fail = []
        for pred_idx, pred in enumerate(sorted_pred_list):
            conf_score = pred[0]
            s_cid, pid, o_cid = pred[1]
            s_tididx, o_tididx = pred[2]
            s_score, p_score, o_score = pred[3]
            s_traj = trajs[s_tididx]
            o_traj = trajs[o_tididx]
            self.add_node(segment, start_id + pred_idx, s_cid, pid, o_cid, s_score, p_score, o_score, s_traj, o_traj, conf_score, success, fail, gt, word_vectors)
        #print('out1')
        self.update_leaf(segment, success)
        #print('out2')
        self.pruning_leaf(vrelation_list, gt)
        #print('out3')
        return fail

    def add_node(self, segment, id, s_cid, pid, o_cid, s_score, p_score, o_score, s_traj, o_traj, conf_score, success, fail, gt, word_vectors, score_thres=0.5):
        rel_score_list = []
        for leaf in self.leaflist:
            s = self.nodes[leaf].s
            p = self.nodes[leaf].p
            o = self.nodes[leaf].o
            
            #############################################
            '''
            if s != s_cid or o != o_cid:
                fail.append(id)
                return
            
            leaf_p_wv = word_vectors[gt.index2pred_name[p]]
            new_p_wv = word_vectors[gt.index2pred_name[pid]]
            
            p_cos_sim = float(np.dot(leaf_p_wv, new_p_wv) / (np.linalg.norm(leaf_p_wv) * np.linalg.norm(new_p_wv)))
            '''
            ###############################################
            
            if s == s_cid and o == o_cid and p == pid:
            #if p == pid and word_vectors.similarity(leaf_s_str, new_s_str) > wv_sim_thres and word_vectors.similarity(leaf_o_str, new_o_str) > wv_sim_thres:
            #if p_cos_sim >= 0.8:
                leaf_s_traj = Trajectory(0, 0, deque(), 0, 0, [])
                leaf_s_traj.copy(self.nodes[leaf].s_traj)
                leaf_s_traj.pstart = s_traj.pstart - 15
                leaf_s_traj.pend = s_traj.pstart + 15
                                    
                leaf_o_traj = Trajectory(0, 0, deque(), 0, 0, [])
                leaf_o_traj.copy(self.nodes[leaf].o_traj)
                leaf_o_traj.pstart = o_traj.pstart - 15
                leaf_o_traj.pend = o_traj.pstart + 15
                                    
                s_viou = association._traj_iou(leaf_s_traj, s_traj)
                o_viou = association._traj_iou(leaf_o_traj, o_traj)
                s_con_score = 0.6 * s_viou + 0.4 * s_score   ##### parameters
                o_con_score = 0.6 * o_viou + 0.4 * o_score
                #s_con_score = s_viou
                #o_con_score = o_viou
                
                if s_con_score > score_thres and o_con_score > score_thres:
                    #rel_score = (s_con_score + o_con_score)/2 + 0.3 * p_score
                    rel_score = (s_con_score + o_con_score + 0.6 * p_score) / 2.6   #############################
                    
                    #total_score = self.sum_score(leaf)
                    path_len = self.path_len(leaf)
                    leaf_path_conf_score, leaf_path_rel_score, _ = self.path_score(leaf)
                    link_path_score = (leaf_path_rel_score*path_len+rel_score)/(path_len+1) # + max(leaf_path_conf_score,conf_score)
                    rel_score_list.append((leaf, conf_score, rel_score, link_path_score))
                    
            else:
                fail.append(id)
                return
            
        if len(rel_score_list) == 0:
            fail.append(id)
        else:
            rel_score_list.sort(key = lambda score:(score[3]), reverse = True)
            leaf = rel_score_list[0][0]
            new_node = Node(
                    id, segment, leaf, [], s_cid, pid, o_cid, s_score, p_score, o_score, s_traj, o_traj, (rel_score_list[0][1], rel_score_list[0][2]))
            self.nodes[id] = new_node
            self.node_num += 1
            self.nodes[leaf].child_id.append(id)
            success.append(id)

    def pruning_leaf(self, vrelation_list, gt, leaf_thres = 5):
        # Control the size of the tree after each segment
        leaf_num = len(self.leaflist)
        #print(leaf_num)
        if leaf_num <= leaf_thres:
            return
        paths = []
        for leaf in self.leaflist:
            #print(leaf)
            _, _, path_score = self.path_score(leaf)
            path = (leaf, path_score)
            #print(path)
            paths.append(path)
        paths.sort(key=lambda p:(p[1]), reverse=True)   ###########################
        
        for leaf, _ in paths[leaf_thres: ]:
            self.leaflist.remove(leaf)
            # Find the nearest separate node
            child = leaf
            parent = self.nodes[child].parent_id
            while len(self.nodes[parent].child_id) == 1:
                child = parent
                parent = self.nodes[child].parent_id
            self.nodes[parent].child_id.remove(child)
            self.nodes[child].parent_id = None
            
            #vrelation = self.generate_vrelation(leaf, gt)
            #vrelation_list.append(vrelation)
            

    def N_scan_pruning(self, id, N = 2):
        parent = id
        for i in range(0, N):
            good_child = parent
            parent = self.nodes[good_child].parent_id
            #print(i, good_child, parent)
            if parent == -1:
                return
        bad_child = []
        bad_child.extend(self.nodes[parent].child_id)
        bad_child.remove(good_child)
        for ch in bad_child:
            self.nodes[parent].child_id.remove(ch)
            self.nodes[ch].parent_id = None
        # pruning from leaflist
        bad_leaf = []
        for leaf in self.leaflist:
            path = self.track_path(leaf)
            for ch in bad_child:
                if ch in path:
                    bad_leaf.append(leaf)
                    break
        for leaf in bad_leaf:
            self.leaflist.remove(leaf)

    def track_path(self, id):
        """Track a path from a leaf node"""
        path = []
        i = id
        #print(i, self.nodes[i].parent_id)
        while i != None and i != -1:
            path.append(i)
            i = self.nodes[i].parent_id
        return path

    def path_most_subobj(self, id):
        path = self.track_path(id)
        sub = []
        obj = []
        for i in path:
            sub.append(self.nodes[i].s)
            obj.append(self.nodes[i].o)
        path_sub = Counter(sub).most_common(1)[0][0]
        path_obj = Counter(obj).most_common(1)[0][0]
        return path_sub, path_obj

    def path_most_pred(self, id):
        path = self.track_path(id)
        pred = []
        for i in path:
            pred.append(self.nodes[i].p)
        path_pred = Counter(pred).most_common(1)[0][0]
        return path_pred

    def path_len(self, id):
        """Return a path's length"""
        path = self.track_path(id)
        return len(path)

    def sum_score(self, id):
        """Return the sum of scores in a path"""
        path = self.track_path(id)
        total_score = 0
        for i in path:
            total_score += self.nodes[i].rel_score[1]
        return total_score
    
    def path_score(self, id):
        """Return a path's score(the max node score in the path, the average node score in the path)"""
        path = self.track_path(id)
        l = len(path)
        total_score = 0
        path_conf_score = 0
        for i in path:
            (conf_score, rel_score) = self.nodes[i].rel_score
            total_score += rel_score
            if conf_score > path_conf_score:
                path_conf_score = conf_score
        path_rel_score = total_score/l
        path_score = path_rel_score # + path_conf_score          ################################
        return path_conf_score, path_rel_score, path_score
        
    def max_node_spo_score(self, id):
        path = self.track_path(id)
        idx = -1
        max_score = 0
        for i in path:
            (conf_score, rel_score) = self.nodes[i].rel_score
            if conf_score > max_score:
                max_score = conf_score
                idx = i
        return self.nodes[idx].s_score, self.nodes[idx].p_score, self.nodes[idx].o_score

    def path_duration(self, id):
        """Return the duration of a path"""
        path = self.track_path(id)
        l = len(path)
        start_node = path[l - 1]
        start = self.nodes[start_node].s_traj.pstart
        end = self.nodes[id].s_traj.pend
        return start, end

    def max_score_path(self):
        """Return the leaf id of the path with max score"""
        paths = []
        for leaf in self.leaflist:
            _, _, path_score = self.path_score(leaf)
            path = (leaf, path_score)
            paths.append(path)
        paths.sort(key=lambda p:(p[1]), reverse=True)       #######################
        return paths[0][0]

    def update_leaf(self, segment, success):
        """Update the leaflist after adding a segment of nodes"""
        leaflist_temp = []
        leaflist_temp.extend(self.leaflist)
        for leaf in leaflist_temp:
            if len(self.nodes[leaf].child_id) != 0:
                self.leaflist.remove(leaf)
        for id in success:
            self.leaflist.append(id)

    def generate_traj(self, leaf):
        """Generate the trajectory of a path"""
        path = self.track_path(leaf)
        path.reverse()
        
        sub_traj = Trajectory(0, 0, deque(), 0, 0, [])
        sub_traj.copy(self.nodes[path[0]].s_traj)
        
        obj_traj = Trajectory(0, 0, deque(), 0, 0, [])
        obj_traj.copy(self.nodes[path[0]].o_traj)
        
        #segment = self.nodes[path[0]].segment
        for id in path[1:]:
            s_traj = Trajectory(0, 0, deque(), 0, 0, [])
            o_traj = Trajectory(0, 0, deque(), 0, 0, [])
            s_traj.copy(self.nodes[id].s_traj)
            o_traj.copy(self.nodes[id].o_traj)
            overlap_length = sub_traj.pend - s_traj.pstart
            
            if overlap_length >= 0:
                sub_traj = association._merge_trajs(sub_traj, s_traj)
                obj_traj = association._merge_trajs(obj_traj, o_traj)
            else:
                missing_length = -overlap_length
                
                s_roi_1 = sub_traj.rois[sub_traj.length() - 1]
                s_roi_2 = s_traj.rois[0]
                o_roi_1 = obj_traj.rois[obj_traj.length() - 1]
                o_roi_2 = o_traj.rois[0]
                for i in range(1, missing_length+1):
                    s_left = (s_roi_2.left() - s_roi_1.left()) / (missing_length + 1) * i + s_roi_1.left()
                    s_top = (s_roi_2.top() - s_roi_1.top()) / (missing_length + 1) * i + s_roi_1.top()
                    s_right = (s_roi_2.right() - s_roi_1.right()) / (missing_length + 1) * i + s_roi_1.right()
                    s_bottom = (s_roi_2.bottom() - s_roi_1.bottom()) / (missing_length + 1) * i + s_roi_1.bottom()
                    sub_traj.predict(drectangle(s_left, s_top, s_right, s_bottom))

                    o_left = (o_roi_2.left() - o_roi_1.left()) / (missing_length + 1) * i + o_roi_1.left()
                    o_top = (o_roi_2.top() - o_roi_1.top()) / (missing_length + 1) * i + o_roi_1.top()
                    o_right = (o_roi_2.right() - o_roi_1.right()) / (missing_length + 1) * i + o_roi_1.right()
                    o_bottom = (o_roi_2.bottom() - o_roi_1.bottom()) / (missing_length + 1) * i + o_roi_1.bottom()
                    obj_traj.predict(drectangle(o_left, o_top, o_right, o_bottom))
                for i in range(s_traj.length()):
                    sub_traj.predict(s_traj.rois[i])
                    obj_traj.predict(o_traj.rois[i])
            #print("end", sub_traj.pstart, sub_traj.pend, sub_traj.score)
        return sub_traj, obj_traj


    def generate_vrelation(self, leaf, gt):
        vrelation = dict()
        
        #path_sub, path_obj = self.path_most_subobj(leaf)
        #path_pred = self.path_most_pred(leaf)
        vrelation['triplet'] = [
            gt.get_object_name(self.nodes[leaf].s),
            gt.get_predicate_name(self.nodes[leaf].p), #########################
            gt.get_object_name(self.nodes[leaf].o)
        ]
        
        vrelation['score'] = float(self.path_score(leaf)[2])
        
        #spo_score = self.max_node_spo_score(leaf)
        #vrelation['spo_score'] = [float(spo_score[0]), float(spo_score[1]), float(spo_score[2])]
        
        start, end = self.path_duration(leaf)
        vrelation['duration'] = [
            int(start),
            int(end)
        ]
        
        sub_traj, obj_traj = self.generate_traj(leaf)
        vrelation['sub_traj'] = sub_traj.serialize()['rois']
        vrelation['obj_traj'] = obj_traj.serialize()['rois']
        return vrelation



'''Operation for treelist'''

def add_segment(vrelation_list, treelist, total_node_num, gt, index, prediction, word_vectors, max_traj_num_in_clip=200):
    vid, fstart, fend = index
    #print("add_segment():")
    #print(index)
    # load prediction data
    pred_list, iou, trackid = prediction
    #pred_list = prediction
    sorted_pred_list = sorted(pred_list, key=lambda x: x[0], reverse=True)
    if len(sorted_pred_list) > max_traj_num_in_clip:
        sorted_pred_list = sorted_pred_list[0:max_traj_num_in_clip]
    # load predict trajectory data
    
    #trajs = object_trajectory_proposal(vid, fstart, fend)
    trajs = object_trajectory_proposal(gt, vid, fstart, fend)
    
    ########using gt s/o label for test#########
    '''
    trajs_gt = object_trajectory_proposal(gt, vid, fstart, fend, gt = True)
    
    for idx,tid in enumerate(trackid):
        if tid >= 0:
            gt_start = idx
            break
    gt_trackid = []
    gt_trackid.extend(trackid)
    for idx,tid in enumerate(gt_trackid):
        if tid >= 0:
            gt_start = idx
            break
    for idx,tid in enumerate(gt_trackid[:gt_start]):
        gt_idx = np.argmax(iou[idx][gt_start:])
        gt_trackid[idx] = gt_idx
    
    tp_sorted_pred_list = []
    tp_sorted_pred_list.extend(sorted_pred_list)
    for idx in range(len(tp_sorted_pred_list)):
        gt_sub = trajs_gt[gt_trackid[tp_sorted_pred_list[idx][2][0]]].category
        gt_obj = trajs_gt[gt_trackid[tp_sorted_pred_list[idx][2][1]]].category
        if gt_sub != tp_sorted_pred_list[idx][1][0] or gt_obj != tp_sorted_pred_list[idx][1][2]:
            sorted_pred_list.remove(tp_sorted_pred_list[idx])
        #triplet = (gt_sub, sorted_pred_list[idx][1][1], gt_obj)
        #sorted_pred_list[idx] = (sorted_pred_list[idx][0], triplet, sorted_pred_list[idx][2])
    '''
    ###########################################
    
    ########using gt tracklet for test#########
    '''
    trajs = object_trajectory_proposal(gt, vid, fstart, fend, gt = True)
    
    for idx,tid in enumerate(trackid):
        if tid >= 0:
            gt_start = idx
            break
    for idx,tid in enumerate(trackid[:gt_start]):
        gt_idx = np.argmax(iou[idx][gt_start:])
        #trackid[idx] = trackid[gt_idx + gt_start]
        trackid[idx] = gt_idx
        assert(trajs[gt_idx].gt_trackid == trackid[gt_idx + gt_start])
        
    for idx in range(len(sorted_pred_list)):
        sorted_pred_list[idx] = (sorted_pred_list[idx][0], sorted_pred_list[idx][1], (trackid[sorted_pred_list[idx][2][0]], trackid[sorted_pred_list[idx][2][1]]))
    '''
    ###########################################
    
    for traj in trajs:
        traj.pstart = fstart
        traj.pend = fend
        traj.vsig = get_segment_signature(vid, fstart, fend)
    # merge
    started_at = time.time()

    segment = fstart
    num = len(sorted_pred_list)
    start_id = total_node_num
    total_node_num += num
    #print("start_id=%d, num=%d"%(start_id, num))
    
    faillist = []
    for tree in treelist:
        #print("in1")
        fail = tree.add_segment(vrelation_list, gt, segment, start_id, num, sorted_pred_list, trajs, word_vectors)
        #print("in2")
        faillist.extend(fail)
    #print('out')
    # Build new tree for all-fail nodes
    tree_num = len(treelist)
    for id in range(start_id, start_id + num):
        if faillist.count(id) == tree_num:
            i = id - start_id
            pred = sorted_pred_list[i]
            conf_score = pred[0]
            s_cid, pid, o_cid = pred[1]
            s_tididx, o_tididx = pred[2]
            s_score, p_score, o_score = pred[3]
            s_traj = trajs[s_tididx]
            o_traj = trajs[o_tididx]
            build_tree(treelist, segment, id, s_cid, pid, o_cid, s_score, p_score, o_score, s_traj, o_traj, conf_score)
    return total_node_num

def build_tree(treelist, segment, id, s_cid, pid, o_cid, s_score, p_score, o_score, s_traj, o_traj, conf_score):
    new_tree = Tree(segment, id, s_cid, pid, o_cid, s_score, p_score, o_score, s_traj, o_traj, conf_score)
    treelist.append(new_tree)

def path_conflict(path1, path2):
    l1 = len(path1)
    l2 = len(path2)
    l = max(l1, l2)
    for i in range(0, l):
        if path1[i] == path2[i]:
            return True
    return False

def select(treelist):
    '''
    max_leaves = dict()
    for tree_id, tree in enumerate(treelist):
        max_leaves[tree_id] = tree.max_score_path()
    return max_leaves
    '''
    max_leaves = dict()
    paths = []
    tree_id_record = []
    node_id_record = []
    conflict_dict = defaultdict(list)
    for tree_id, tree in enumerate(treelist):
        for leaf in tree.leaflist:
            _, _, path_score = tree.path_score(leaf)
            paths.append((tree_id, tree.track_path(leaf), path_score))
    paths.sort(key = lambda path:(path[2]), reverse = True)        ##########################
    for path in paths:
        if path[0] not in tree_id_record:
            conflict = -1
            path[1].reverse()
            for i, node in enumerate(path[1]):
                if node in node_id_record:
                    conflict = i
                    break
            if conflict == -1:
                max_leaves[path[0]] = path[1][-1]
                tree_id_record.append(path[0])
                node_id_record.extend(path[1])
            else:
                if path[0] not in conflict_dict.keys():
                    conflict_dict[path[0]].append(path[1][conflict - 1])
        
    for tree_id, tree in enumerate(treelist):
        if tree_id not in tree_id_record:
            tree.leaflist = []
            tree.leaflist.extend(conflict_dict[tree_id])
            
            #max_path_score = 0
            paths = []
            for leaf in tree.leaflist:
                tree.nodes[leaf].child_id = []
                _, _, path_score = tree.path_score(leaf)
                paths.append((leaf, path_score))
                #if path_score > max_path_score:
                    #max_path_score = path_score
                    #max_leaves[tree_id] = leaf
            paths.sort(key = lambda path:(path[1]), reverse = True)    ################################
            max_leaves[tree_id] = paths[0][0]
            tree_id_record.append(tree_id)
            
    return max_leaves
    
def pruning(treelist, max_leaves):
    for i, tree in enumerate(treelist):
        tree.N_scan_pruning(max_leaves[i])

def check_trees(treelist, vrelation_list, segment, max_leaves, gt, max_node_gap=45, min_path_len=1):
    """
    Find the tree that haven't been updated for a time
    and let the path with max score in it be a track result
    and remove this tree
    """
    treelist_temp = []
    treelist_temp.extend(treelist)
    for tree_id, tree in enumerate(treelist_temp):
        max_seg = 0
        for leaf in tree.leaflist:
            if tree.nodes[leaf].s_traj.pend > max_seg:
                max_seg = tree.nodes[leaf].s_traj.pend
        if max_seg <= segment - max_node_gap:
            max_leaf = max_leaves[tree_id]
            if tree.path_len(max_leaf) >= min_path_len: # According to gt, vrelation can have just 1 node
                vrelation = tree.generate_vrelation(max_leaf, gt)
                vrelation_list.append(vrelation)
            treelist.remove(tree)

def generate_results(treelist, vrelation_list, max_leaves, gt):
    """Generate the association results"""
    for tree_id, tree in enumerate(treelist):
        #print(tree.path_len(max_leaves[tree_id]))
        vrelation = tree.generate_vrelation(max_leaves[tree_id], gt)
        vrelation_list.append(vrelation)

"""Total Association Operation for a video"""
def mht_association(gt, short_term_relations, word_vectors):
    treelist = []
    total_node_num = 0
    vrelation_list = []

    short_term_relations.sort(key=lambda x: int(x[0][1]))
    max_leaves = dict()
    for i, (index, prediction) in enumerate(short_term_relations):
        segment = index[1]
        #print("segment="+str(segment))
        check_trees(treelist, vrelation_list, segment, max_leaves, gt)
        total_node_num = add_segment(vrelation_list, treelist, total_node_num, gt, index, prediction, word_vectors)
        max_leaves = select(treelist)
        #print(max_leaves)
        pruning(treelist, max_leaves)
        #check_trees(treelist, vrelation_list, segment, gt)
        
    #print("/nGenerate_total_results/n")
    generate_results(treelist, vrelation_list, max_leaves, gt)

    return vrelation_list
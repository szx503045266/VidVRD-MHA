import os
import pickle
import argparse
import itertools
import time
from collections import deque, defaultdict
from copy import deepcopy

from dlib import drectangle
import numpy as np

from baseline import *
from baseline.association import *
from baseline.trajectory import *

viou_thres = 0.5
s_score_thres = 0.5
o_score_thres = 0.5
rel_score_thres = 0.5

class Node(object):
    
    def __init__(self, id, segment, parent_id, child_id, s, p, o, s_traj, o_traj, rel_score):
        # child_id is a list
        self.id = id
        self.segment = segment
        self.parent_id = parent_id
        self.child_id = child_id
        self.s = s
        self.p = p
        self.o = o
        self.s_traj = s_traj
        self.o_traj = o_traj
        self.rel_score = rel_score

class Tree(object):

    def __init__(self, segment, id, s, p, o, s_traj, o_traj, rel_score):
        self.nodes = dict()
        self.nodes[id] = Node(id, segment, -1, [], s, p, o, s_traj, o_traj, rel_score)
        self.node_num = 1
        self.leaflist = [id]

    def add_segment(self, vrelation_list, dataset, segment, start_id, num, sorted_pred_list, trajs):
        '''The input are rel_feat, s_traj, o_traj of all tracklets in this segment.
        Return the index of the node not linked'''
        success = []
        fail = []
        for pred_idx, pred in enumerate(sorted_pred_list):
            conf_score = pred[0]
            s_cid, pid, o_cid = pred[1]
            s_tididx, o_tididx = pred[2]
            s_traj = trajs[s_tididx]
            o_traj = trajs[o_tididx]
            self.add_node(segment, start_id + pred_idx, s_cid, pid, o_cid, s_traj, o_traj, conf_score, success, fail)
        #print('out1')
        self.update_leaf(segment, success)
        #print('out2')
        self.pruning_leaf(vrelation_list, dataset)
        #print('out3')
        return fail

    def add_node(self, segment, id, s_cid, pid, o_cid, s_traj, o_traj, conf_score, success, fail, score_thres=0.5):
        add = False
        rel_score_list = []
        for leaf in self.leaflist:
            rel_score = self.nodes[leaf].rel_score
            rel_score_list.append(rel_score)
        # Select the leaf which has the max connection rel_score with node id
        leaf_id = rel_score_list.index(max(rel_score_list))
        leaf = self.leaflist[leaf_id]
        
        s = self.nodes[leaf].s
        p = self.nodes[leaf].p
        o = self.nodes[leaf].o
        
        if s == s_cid and o == o_cid and p == pid:
            
            leaf_s_traj = Trajectory(0, 0, deque(), 0, 0, 0)
            leaf_s_traj.copy(self.nodes[leaf].s_traj)
            leaf_s_traj.pstart = s_traj.pstart - 15
            leaf_s_traj.pend = s_traj.pstart + 15
                                
            leaf_o_traj = Trajectory(0, 0, deque(), 0, 0, 0)
            leaf_o_traj.copy(self.nodes[leaf].o_traj)
            leaf_o_traj.pstart = o_traj.pstart - 15
            leaf_o_traj.pend = o_traj.pstart + 15
                                
            s_viou = association._traj_iou(leaf_s_traj, s_traj)
            o_viou = association._traj_iou(leaf_o_traj, o_traj)
            s_score = 0.5 * s_viou + 0.5 * conf_score/10
            o_score = 0.5 * o_viou + 0.5 * conf_score/10
            #s_score = s_viou
            #o_score = o_viou
            if s_score > score_thres and o_score > score_thres:
            
                new_node = Node(
                        id, segment, leaf, [], s_cid, pid, o_cid, s_traj, o_traj, conf_score)
                #print(id, leaf)
                self.nodes[id] = new_node
                self.node_num += 1
                self.nodes[leaf].child_id.append(id)
                add = True

        if add == False:
            fail.append(id)
        else:
            success.append(id)

    def pruning_leaf(self, vrelation_list, dataset, leaf_thres = 1):
        # Control the size of the tree after each segment
        leaf_num = len(self.leaflist)
        #print(leaf_num)
        if leaf_num <= leaf_thres:
            return
        paths = []
        for leaf in self.leaflist:
            #print(leaf)
            path = (leaf, self.track_path(leaf), self.path_score(leaf))
            #print(path)
            paths.append(path)
        paths.sort(key=lambda p:p[2], reverse=True)
        
        for leaf, _, _ in paths[leaf_thres: ]:
            self.leaflist.remove(leaf)
            # Find the nearest separate node
            child = leaf
            parent = self.nodes[child].parent_id
            while len(self.nodes[parent].child_id) == 1:
                child = parent
                parent = self.nodes[child].parent_id
            self.nodes[parent].child_id.remove(child)
            self.nodes[child].parent_id = None
            
            #vrelation = self.generate_vrelation(leaf, dataset)
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

    def path_len(self, id):
        """Return a path's length"""
        path = self.track_path(id)
        return len(path)

    def path_score(self, leaf):
        """Return a path's score"""
        path = self.track_path(leaf)
        l = len(path)
        total_score = 0
        for i in path:
            total_score += self.nodes[i].rel_score
        return total_score/l

    def path_duration(self, leaf):
        """Return the duration of a path"""
        path = self.track_path(leaf)
        l = len(path)
        start_node = path[l - 1]
        start = self.nodes[start_node].s_traj.pstart
        end = self.nodes[leaf].s_traj.pend
        return start, end

    def max_score_path(self):
        """Return the leaf id of the path with max score"""
        paths = []
        for leaf in self.leaflist:
            path = (leaf, self.path_score(leaf))
            paths.append(path)
        paths.sort(key=lambda p:p[1], reverse=True)
        return paths[0][0]

    def update_leaf(self, segment, success):
        """Update the leaflist after adding a segment of nodes"""
        leaflist_temp = []  #fix a bug
        leaflist_temp.extend(self.leaflist)
        for leaf in leaflist_temp:
            if len(self.nodes[leaf].child_id) != 0:
                self.leaflist.remove(leaf)
        for id in success:
            self.leaflist.append(id)

    def score(self, leaf, id, s_feat, o_feat, rel_feat, s_traj, o_traj):
        '''Return the leaf node info (3 score) with max linking rel_score'''
        #TODO
        return 0.5, 0.5, 0.5

    def generate_traj(self, leaf):
        """Generate the trajectory of a path"""
        path = self.track_path(leaf)
        path.reverse()
        #if(leaf == 180 or leaf == 182 or leaf == 188):
            #print()
            #print(path)
        #print(path)
        
        sub_traj = Trajectory(0, 0, deque(), 0, 0, 0)
        #if(leaf == 180 or leaf == 182 or leaf == 188):
            #print("b1_start", path[0], len(self.nodes[path[0]].s_traj.rois))
        sub_traj.copy(self.nodes[path[0]].s_traj)
        #if(leaf == 180 or leaf == 182 or leaf == 188):
            #print("b2_start", len(sub_traj.rois), "(%d,%d)"%(sub_traj.pstart, sub_traj.pend))
        
        obj_traj = Trajectory(0, 0, deque(), 0, 0, 0)
        obj_traj.copy(self.nodes[path[0]].o_traj)
        
        #segment = self.nodes[path[0]].segment
        for id in path[1:]:
            s_traj = Trajectory(0, 0, deque(), 0, 0, 0)
            o_traj = Trajectory(0, 0, deque(), 0, 0, 0)
            s_traj.copy(self.nodes[id].s_traj)
            o_traj.copy(self.nodes[id].o_traj)
            #if(leaf == 180 or leaf == 182 or leaf == 188):
                #print(id, len(sub_traj.rois), len(s_traj.rois))
                #print("(%d,%d)(%d,%d)"%(sub_traj.pstart, sub_traj.pend, s_traj.pstart, s_traj.pend))
            overlap_length = sub_traj.pend - s_traj.pstart
            
            if overlap_length >= 0:
                #print("overlap_length=%d"%overlap_length)
                #if(leaf == 180 or leaf == 182 or leaf == 188):
                    #print(len(sub_traj.rois), "(%d,%d)"%(sub_traj.pstart,sub_traj.pend), sub_traj.rois[15], s_traj.rois[0])
                sub_traj = association._merge_trajs(sub_traj, s_traj)
                obj_traj = association._merge_trajs(obj_traj, o_traj)
                #if(leaf == 180 or leaf == 182 or leaf == 188):
                    #print(len(sub_traj.rois), "(%d,%d)"%(sub_traj.pstart,sub_traj.pend), sub_traj.rois[15])
            else:
                missing_length = -overlap_length
                
                s_roi_1 = sub_traj.rois[sub_traj.length() - 1]
                s_roi_2 = s_traj.rois[0]
                o_roi_1 = obj_traj.rois[obj_traj.length() - 1]
                o_roi_2 = o_traj.rois[0]
                for i in range(1, missing_length+1):
                    s_left = (s_roi_1.left() + s_roi_2.left()) / (missing_length + 1) * i
                    s_top = (s_roi_1.top() + s_roi_2.top()) / (missing_length + 1) * i
                    s_right = (s_roi_1.right() + s_roi_2.right()) / (missing_length + 1) * i
                    s_bottom = (s_roi_1.bottom() + s_roi_2.bottom()) / (missing_length + 1) * i
                    sub_traj.predict(drectangle(s_left, s_top, s_right, s_bottom))

                    o_left = (o_roi_1.left() + o_roi_2.left()) / (missing_length + 1) * i
                    o_top = (o_roi_1.top() + o_roi_2.top()) / (missing_length + 1) * i
                    o_right = (o_roi_1.right() + o_roi_2.right()) / (missing_length + 1) * i
                    o_bottom = (o_roi_1.bottom() + o_roi_2.bottom()) / (missing_length + 1) * i
                    obj_traj.predict(drectangle(o_left, o_top, o_right, o_bottom))
                for i in range(s_traj.length()):
                    sub_traj.predict(s_traj.rois[i])
                    obj_traj.predict(o_traj.rois[i])
            #print("end", sub_traj.pstart, sub_traj.pend, sub_traj.score)
        return sub_traj, obj_traj


    def generate_vrelation(self, leaf, dataset):
        vrelation = dict()
        vrelation['triplet'] = [
            dataset.get_object_name(self.nodes[leaf].s),
            dataset.get_predicate_name(self.nodes[leaf].p),
            dataset.get_object_name(self.nodes[leaf].o)
        ]
        vrelation['score'] = float(self.path_score(leaf))
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

def add_segment(vrelation_list, treelist, total_node_num, dataset, index, prediction, max_traj_num_in_clip=100):
    vid, fstart, fend = index
    #print("add_segment():")
    #print(index)
    # load prediction data
    pred_list, iou, trackid = prediction
    sorted_pred_list = sorted(pred_list, key=lambda x: x[0], reverse=True)
    if len(sorted_pred_list) > max_traj_num_in_clip:
        sorted_pred_list = sorted_pred_list[0:max_traj_num_in_clip]
    # load predict trajectory data
    trajs = object_trajectory_proposal(dataset, vid, fstart, fend)
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
        fail = tree.add_segment(vrelation_list, dataset, segment, start_id, num, sorted_pred_list, trajs)
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
            s_traj = trajs[s_tididx]
            o_traj = trajs[o_tididx]
            build_tree(treelist, segment, id, s_cid, pid, o_cid, s_traj, o_traj, conf_score)
    return total_node_num

def build_tree(treelist, segment, id, s_cid, pid, o_cid, s_traj, o_traj, conf_score):
    new_tree = Tree(segment, id, s_cid, pid, o_cid, s_traj, o_traj, conf_score)
    treelist.append(new_tree)

def path_conflict(path1, path2):
    l1 = len(path1)
    l2 = len(path2)
    l = max(l1, l2)
    for i in range(0, l):
        if path1[i] == path2[i]:
            return True
    return False

def mwis(treelist):
    max_leaves = []
    for tree in treelist:
        max_leaves.append(tree.max_score_path())
    return max_leaves

def pruning(treelist, max_leaves):
    for i, tree in enumerate(treelist):
        tree.N_scan_pruning(max_leaves[i])

def check_trees(treelist, vrelation_list, segment, dataset, max_node_gap=15, min_path_len=1):
    """
    Find the tree that haven't been updated for a time
    and let the path with max score in it be a track result
    and remove this tree
    """
    treelist_temp = []
    treelist_temp.extend(treelist)
    for tree in treelist_temp:
        max_seg = 0
        for leaf in tree.leaflist:
            if tree.nodes[leaf].s_traj.pend > max_seg:
                max_seg = tree.nodes[leaf].s_traj.pend
        if max_seg <= segment - max_node_gap:
            max_leaf = tree.max_score_path()
            if tree.path_len(max_leaf) >= min_path_len: # Too short path will not be selected to be a vrelation
                vrelation = tree.generate_vrelation(max_leaf, dataset)
                vrelation_list.append(vrelation)
            treelist.remove(tree)

def generate_results(treelist, vrelation_list, dataset):
    """Generate the association results"""
    for tree in treelist:
        #print('tree')
        max_leaf = tree.max_score_path()
        vrelation = tree.generate_vrelation(max_leaf, dataset)
        vrelation_list.append(vrelation)

"""Total Association Operation for a video"""
def mht_association(dataset, short_term_relations):
    treelist = []
    total_node_num = 0
    vrelation_list = []

    short_term_relations.sort(key=lambda x: int(x[0][1]))
    for i, (index, prediction) in enumerate(short_term_relations):
        segment = index[1]
        #print("segment="+str(segment))
        check_trees(treelist, vrelation_list, segment, dataset)
        total_node_num = add_segment(vrelation_list, treelist, total_node_num, dataset, index, prediction)
        max_leaves = mwis(treelist)
        pruning(treelist, max_leaves)
        #check_trees(treelist, vrelation_list, segment, dataset)
        
    #print("/nGenerate_total_results/n")
    generate_results(treelist, vrelation_list, dataset)

    return vrelation_list
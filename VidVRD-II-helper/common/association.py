import copy
from collections import defaultdict

import numpy as np

from .relation import VideoRelation


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
                    last_r.extend(this_r)
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

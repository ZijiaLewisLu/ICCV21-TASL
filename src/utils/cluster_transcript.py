import numpy as np
import json

def edit_distance(a, b):
    distance_dict = {}
    la, lb = len(a), len(b)
    return _dist_helper(a, b, la, lb, distance_dict)
    
def _dist_helper(a, b, la, lb, distance_dict):
    if (la, lb) not in distance_dict:
        if la == 0:
            score = lb
        elif lb == 0:
            score = la
        else:
            score1 = _dist_helper(a, b, la-1, lb, distance_dict) + 1
            score2 = _dist_helper(a, b, la, lb-1, distance_dict) + 1
    
            score3 = _dist_helper(a, b, la-1, lb-1, distance_dict)
            score3 = score3 if a[la-1] == b[lb-1] else score3 + 1
            
            score = min([score1, score2, score3])
        distance_dict[(la, lb)] = score
        
    return distance_dict[(la, lb)]

def compute_distance_matrix(trans_list):
    N = len(trans_list)
    abs_dist = np.zeros([N, N])
    rel_dist = np.zeros([N, N])
    for i in range(N):
        for j in range(i+1, N):
            dist = edit_distance(trans_list[i], trans_list[j])
            abs_dist[i, j] = abs_dist[j, i] = dist
            dist = 2 * dist / (len(trans_list[i]) + len(trans_list[j]))
            rel_dist[i, j] = rel_dist[j, i] = dist
    return abs_dist, rel_dist

def compute_set_score(score_matrix, subset):
    subset_matrix = score_matrix[:, subset]
    subset_matrix = subset_matrix.max(1)
    
    total_score = subset_matrix.sum()
    return total_score

def greedy_iteration(score_matrix, subset):
    """run one iteration of greedy iteration and update subset"""
    N = score_matrix.shape[0]
    remain_index = [ i for i in range(N) if i not in subset ]
    
    max_score = -np.inf
    max_subset = None
    for i in remain_index:
        new_subset = subset + [i]
        new_score = compute_set_score(score_matrix, new_subset)
        if new_score > max_score:
            max_score = new_score
            max_subset = new_subset
        
    return max_score, max_subset

def subset_selection(score_matrix, k):
    """
    find best representative subset of size k
    Input:
        score_matrix: higher score means larger similarity
    Output:
        subset: indices of representative elements
        assignment: list of indices, show which representative each element is assigned to
        cluster_dict: { representative idx: [ member idx, ... ]  } 
    """ 
    subset = []
    # history = []
    for i in range(k):
        score, subset = greedy_iteration(score_matrix, subset)
        # history.append([score, subset])

    assignment = score_matrix[:, subset]
    assignment = assignment.argmax(1)
    assignment = np.array([ subset[i] for i in assignment ])
    cluster_dict = {}
    for si in subset:
        idxs = np.where(assignment==si)[0]
        cluster_dict[si] = idxs
    return score, subset, assignment, cluster_dict

def cluster_transcript(trans_list: list, num_group: int):
    """
    Input:
        trans_list: list of transcripts
                    each transcript is a list of action name(str), instead of action ids(int)
        num_group: num of group to form
    
    Return:
        grammar_dict: key - representative transcript of a group
                      value - list of transcripts in the group
    """
    trans_str_list = [ " ".join(trans) for trans in trans_list ]
    trans_str_list = list(set(trans_str_list)) # remove duplicate transcripts
    trans_list = [ trans.split(" ") for trans in trans_str_list ]
    
    trans_list_short = []
    for trans in trans_list: # remove background action when computing the distance between transcripts
        new_trans = [ t for t in trans if t != "SIL" ]
        trans_list_short.append(new_trans)

    abs_dist, rel_dist = compute_distance_matrix(trans_list_short)

    score, subset, assignment, cluster_dict = subset_selection(-rel_dist, num_group)

    grammar_dict = {}
    for si, member_indices in cluster_dict.items():
        rep_trans = trans_list_short[si]
        rep_trans = " ".join(rep_trans)
        
        member_trans = [ trans_list[j] for j in member_indices ]
        
        grammar_dict[rep_trans] = member_trans
    
    return grammar_dict

if __name__ == '__main__':
    train_trans_list = '...'
    grammar_dict = cluster_transcript(train_trans_list, 20)
    with open('transcript_20cluster.json', 'w') as fp:
        json.dump(grammar_dict, fp)
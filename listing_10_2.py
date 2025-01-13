import pandas as pd
import numpy as np

def get_random_subspaces(features_arr, num_base_detectors, num_feats_per_detector):
    num_feats = len(features_arr)
    feat_sets_arr = []
    ft_used_counts = np.zeros(num_feats)
    ft_pair_mtx = np.zeros((num_feats, num_feats))

    for _ in range(num_base_detectors):
        min_count = ft_used_counts.min()
        idxs = np.where(ft_used_counts == min_count)[0]
        
        feat_set = [np.random.choice(idxs)]

        while len(feat_set) < num_feats_per_detector:
            mtx_with_set = ft_pair_mtx[:, feat_set]
            sums = mtx_with_set.sum(axis=1)
            min_sum = sums.min()
            min_idxs = np.where(sums == min_sum)[0]
            new_feat = np.random.choice(min_idxs)
            feat_set.append(new_feat)
            feat_set = list(set(feat_set))

            for c in feat_set:
                ft_pair_mtx[c][new_feat] += 1
                ft_pair_mtx[new_feat][c] += 1
        
        for c in feat_set:
            ft_used_counts[c] += 1
        
        feat_sets_arr.append(feat_set)

    return feat_sets_arr

np.random.seed(0)
features_arr = ["A", "B", "C", "D", "E", "F", "G", "H"]
num_base_detectors = 4
num_feats_per_detector = 5

feat_sets_arr = get_random_subspaces(features_arr, num_base_detectors, num_feats_per_detector)
for feat_set in feat_sets_arr:
    print([features_arr[x] for x in feat_set])
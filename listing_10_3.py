#Não funciona
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml

def highest_corr():
    return np.unravel_index(
        np.argmax(corr_matrix.values, axis=None)
        , corr_matrix.shape
    )

def get_correlated_subspaces(corr_matrix, num_base_detectors, num_feats_per_detector):

    sets = []
    for _ in range(num_base_detectors):
        m1, m2 = highest_corr()
        curr_set = [m1, m2]
        for _ in range(2, num_feats_per_detector):
            m = np.unravel_index(
                np.argsort(corr_matrix.values, axis=None),
                corr_matrix.shape
            )
            m0 = m[0][::1]
            m1 = m[1][::1]
            for i in range(len(m0)):
                d0 = m0[i]
                d1 = m1[i]
                if (d0 in curr_set) or (d1 in curr_set):
                    curr_set.append(d0)
                    curr_set = list(set(curr_set))
                    if len(curr_set) < num_feats_per_detector:
                        curr_set.append(d1)
                        curr_set = list(set(curr_set))
                if len(curr_set) >= num_feats_per_detector:
                    break
            for i in curr_set:
                i_idx = corr_matrix.index[i]
                for j in curr_set:
                    j_idx = corr_matrix.columns[j]
                    corr_matrix.loc[i_idx, j_idx] = 0
            if len(curr_set) >= num_feats_per_detector:
                break
        sets.append(curr_set)
    return sets

data = fetch_openml('baseball', version=1)
df = pd.DataFrame(data.data, columns=data.feature_names)
corr_matrix = abs(df.corr(method='spearman'))
corr_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
corr_matrix = corr_matrix.fillna(0)
feat_sets_arr = get_correlated_subspaces(corr_matrix, num_base_detectors=5, 
                                         num_feats_per_detector=4)
for feat_set in feat_sets_arr:    
   print([df.columns[x] for x in feat_set])
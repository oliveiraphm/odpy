import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.covariance import MinCovDet
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from listing_7_2 import create_four_clusters_test_data

def cluster_based_outliers(df, min_n_clusters=1, max_n_clusters=20,
                           n_trials=10):
    scores_df = pd.DataFrame()
    scores_col_name = 0
    X = df.copy()
    for n_clusters in range(min_n_clusters, max_n_clusters+1): 
        for trial_idx in range(n_trials): 
            seed = n_clusters * 100 + trial_idx
            np.random.seed(seed)
            kmeans = KMeans(n_clusters=n_clusters, random_state=seed, 
 init='random', n_init="auto").fit(df) 
            X['Cluster ID'] = kmeans.labels_

            dfs_arr = []
            for cluster_idx in range(n_clusters): 
                cluster_df = X[X['Cluster ID'] == cluster_idx].copy()
                kde = KernelDensity(
                    kernel='gaussian').fit(cluster_df[df.columns]) 
                kde_scores = (-1) * \
                   kde.score_samples(cluster_df[df.columns])
                cluster_df['KDE Score'] = kde_scores / kde_scores.mean() 
                dfs_arr.append(cluster_df)

            scores_col_name += 1
            scores_df = pd.concat([scores_df, 
              pd.concat(dfs_arr).sort_index()['KDE Score']], axis=1)
    return scores_df.mean(axis=1)

df = create_four_clusters_test_data()
df = pd.DataFrame(RobustScaler().fit_transform(df), columns=df.columns)
df['Cluster-Based Scores'] = cluster_based_outliers(df)
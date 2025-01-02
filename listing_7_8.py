import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from listing_7_2 import create_four_clusters_test_data

def cluster_based_outliers(df, sizes_weight=0.05):
    clf_if = IsolationForest()
    clf_if.fit(df)
    pred = clf_if.decision_function(df)
    trimmed_df = df.loc[np.argsort(pred)[50:]]

    best_score = np.inf
    best_n_clusters = -1
    for n_clusters in range(2, 10):
        gmm = GaussianMixture(n_components=n_clusters)
        gmm.fit(trimmed_df)
        score = gmm.bic(trimmed_df)
        if score < best_score:
            best_score = score
            best_n_clusters = n_clusters

    gmm = GaussianMixture(n_components=best_n_clusters)
    gmm.fit(trimmed_df)
    X = df.copy()
    X['Cluster ID'] = gmm.predict(df)

    vc = pd.Series(X['Cluster ID']).value_counts()
    cluster_counts_dict = {x: y for x, y in zip(vc.index, vc.values)}
    size_scores = [cluster_counts_dict[x] for x in X['Cluster ID']]
    scaler = MinMaxScaler()
    size_scores = scaler.fit_transform(np.array(size_scores).reshape(-1, 1))
    size_scores = np.array(size_scores).reshape(1,-1)[0]
    size_scores = np.array([1.0 - x for x in size_scores])

    dfs_arr = []
    for cluster_idx in range(best_n_clusters):
        cluster_df = X[X['Cluster ID'] == cluster_idx].copy()
        cov = MinCovDet(random_state=0).fit(cluster_df[df.columns])
        cluster_df['Mahalanobis Dist'] = cov.mahalanobis(cluster_df[df.columns])
        cluster_df['Mahalanobis Dist'] = (cluster_df['Mahalanobis Dist'] / cluster_df['Mahalanobis Dist'].mean())
        dfs_arr.append(cluster_df)
    maha_scores = pd.concat(dfs_arr).sort_index()['Mahalanobis Dist']
    scaler = MinMaxScaler()
    maha_scores = scaler.fit_transform(np.array(maha_scores).reshape(-1, 1))
    maha_scores = np.array(maha_scores).reshape(1,-1)[0]
    return (size_scores * sizes_weight) + maha_scores

df = create_four_clusters_test_data()
df = pd.DataFrame(RobustScaler().fit_transform(df), columns=df.columns)
df['Cluster-Based Scores'] = cluster_based_outliers(df)
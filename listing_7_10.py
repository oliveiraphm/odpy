import pandas as pd
import numpy as np
from kmodes.kmodes import KModes

class kmodesOutlierDetector:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit_predict(self, df):
        km = KModes(n_clusters=self.n_clusters, init='Huang', n_init=5)
        clusters = km.fit_predict(df)
        df_copy = df.copy()
        df_copy['Cluster'] = clusters

        scores = [-1] * len(df)
        for cluster_idx in range(self.n_clusters):
            cluster_df = df_copy[df_copy['Cluster'] == cluster_idx]
            center = km.cluster_centroids_[cluster_idx]
            for i in cluster_df.index:
                row = cluster_df.loc[i]
                num_diff = 0
                for j in range(len(center)):
                    if row[j] != center[j]:
                        num_diff += 1
                scores[i] = num_diff
            return scores
        
    np.random.seed(0)


data = np.random.choice(['A', 'B', 'C'], (100, 2)) 
df = pd.DataFrame(data, columns=['F1', 'F2'])
df['F3'] = df['F2']
df['F4'] = df['F1'] + df['F2']
df.loc[99] = ['A', 'A', 'B', 'CC'] 

clf = kmodesOutlierDetector(n_clusters=9)
scores = clf.fit_predict(df)
df['KModes Scores'] = scores
df.tail()
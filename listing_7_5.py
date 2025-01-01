from listing_7_2 import create_four_clusters_test_data
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from sklearn.preprocessing import RobustScaler

class LDOFOutlierDetector:
    def __initt__(self):
        pass

    def fit_predict(self, df, k):
        balltree = BallTree(df)
        dist, ind = balltree.query(df, k=k)
        knn_distances = [x.mean() for x in dist]

        inner_distances = []
        for i in df.index:
            local_balltree = BallTree(df.loc[ind[i]])
            local_dist, local_ind = balltree.query(df.loc[[i]], k=k)
            inner_distances.append(local_dist.mean())

            return np.array(knn_distances) / np.array(inner_distances)
        
df = create_four_clusters_test_data()
df = pd.DataFrame(RobustScaler().fit_transform(df), columns=df.columns)

clf = LDOFOutlierDetector()
df['LDOF Score'] = clf.fit_predict(df, k=20)
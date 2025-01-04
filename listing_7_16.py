from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from listing_7_2 import create_four_clusters_test_data

class DMLOutlierDetection:
    def __init__(self):
        pass

    def fit_predict(self, df):
        real_df = df.copy()
        real_df['Real'] = True

        synth_df = pd.DataFrame()
        for col_name in df.columns:
            mean = df[col_name].mean()
            sttdev = df[col_name].std()
            synth_df[col_name] = np.random.normal(loc=mean, scale=sttdev, size=len(df))
        synth_df['Real'] = False

        train_df = pd.concat([real_df, synth_df])

        clf = RandomForestClassifier(max_depth=5)
        clf.fit(train_df.drop(columns=['Real']), train_df['Real'])

        r = clf.apply(df)

        scores = [0] * len(df)

        for tree_idx in range(len(r[0])):
            c = Counter(r[:, tree_idx])
            for record_idx in range(len(r)):
                node_idx = r[record_idx, tree_idx]
                node_count = c[node_idx]
                scores[record_idx] += len(df) - node_count

        return scores
    
df = create_four_clusters_test_data()
df = pd.DataFrame(RobustScaler().fit_transform(df), columns=df.columns)
clf = DMLOutlierDetection()
df['Scores'] = clf.fit_predict(df)

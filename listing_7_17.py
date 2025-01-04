import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import BallTree
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from listing_7_2 import create_four_clusters_test_data


class NearestSampleOutlierDetector:
    def __init__(self, n_iterations=10, n_samples=100, 
                 show_progress=False):
        self.n_iterations = n_iterations
        self.n_samples = n_samples
        self.show_progress = show_progress
        self.training_df = None
        self.orig_cols = None
        self.tree = None

    def fit(self, df):
        self.training_df = df.copy()
        self.orig_cols = df.columns
        return self

    def decision_function(self, df_in):
        df = pd.DataFrame(df_in).copy()
        self.balltree = BallTree(df) #A

        if self.show_progress: #B
            for iteration_idx in tqdm(range(self.n_iterations)):
                scores = self.execute_iteration(df)
                self.df[f'Scores_{iteration_idx}'] = scores
        else:
            for iteration_idx in range(self.n_iterations):
                scores = self.execute_iteration(df)
                df[f'Scores_{iteration_idx}'] = scores

        score_cols = [f'Scores_{x}' for x in range(iteration_idx)]
        df['Score'] = df[score_cols].sum(axis=1)

        return df['Score'].values
    
    def execute_iteration(self, prediction_df): #C
        sample_idxs = np.random.choice(range(len(self.training_df)), 
                                       self.n_samples) #D

        distances_arr = []
        for sample_idx in sample_idxs: #E
            row = self.training_df.iloc[sample_idx: sample_idx+1]
            dist, ind = self.balltree.query(row[self.orig_cols],
                          k=len(prediction_df)) #F
            dist = dist[0]
            ind = ind[0]
            ordered_idxs = np.argsort(ind)
            dist = pd.Series(dist)
            distances_arr.append(dist.loc[ordered_idxs].values)

        distances_df = pd.DataFrame(distances_arr).T #G
        return np.array(distances_df.min(axis=1))

df = create_four_clusters_test_data()
df = pd.DataFrame(RobustScaler().fit_transform(df), columns=df.columns)
df['NearestSample Scores'] = NearestSampleOutlierDetector().fit(df).decision_function(df)

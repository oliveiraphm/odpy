from scipy.spatial import ConvexHull
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from listing_7_2 import create_four_clusters_test_data

class ConvexHullOutlierDetector:
    def __init__(self, num_iterations):
        self.num_iterations = num_iterations
    
    def fit_predict(self, df):
        scores = [0] * len(df) #A
        remaining_df = df.copy()
        remaining_df['Row Idx'] = remaining_df.index

        for interation_idx in range(self.num_iterations): #B
            hull = ConvexHull(remaining_df[[df.columns[0], df.columns[1]]]) #C

            simplex_idxs = [y for x in hull.simplices for y in x]
            simplex_idxs = list(set(simplex_idxs))
            for idx in simplex_idxs:
                orig_row_idx = remaining_df.loc[idx, 'Row Idx']
                scores[orig_row_idx] = (self.num_iterations - interation_idx) #D
                remaining_df = remaining_df.drop(index=idx) #E
            remaining_df = remaining_df.reset_index(drop=True)

        return scores
df = create_four_clusters_test_data()
df = pd.DataFrame(RobustScaler().fit_transform(df), columns=df.columns)
df['Convex Hull Scores'] = ConvexHullOutlierDetector(num_iterations=2).fit_predict(df)
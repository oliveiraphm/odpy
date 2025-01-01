from listing_7_2 import create_four_clusters_test_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import BallTree
from collections import Counter

class ODINOutlierDetector:
    def __init__(self):
        pass

    def fit_predict(self, df_in, k):
        df = df_in.copy()
        num_rows = len(df)
        b = BallTree(df)
        dist, ind = b.query(df, k=k)

        df['ODIN Score'] = [0] * num_rows
        for current_k in range(1, k):
            current_ind = ind[:, current_k]
            c = Counter(current_ind)
            g = [(x, c[x]) if x in c else (x, 0) for x in range(num_rows)]
            df['ODIN Score'] += (k - current_k) * np.array([x[1] for x in g])

        min_score = df['ODIN Score'].min()
        max_score = df['ODIN Score'].max()
        return (max_score - df['ODIN Score']) / (max_score - min_score)
    
np.random.seed(0)
df = create_four_clusters_test_data()
clf = ODINOutlierDetector()
df['ODIN Score'] = clf.fit_predict(df, k=20)
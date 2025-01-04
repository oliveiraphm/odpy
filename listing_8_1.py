from sklearn.cluster import KNN
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

scores_arr = []
k_vals = [5, 10, 20, 50, 100]
for k in k_vals:
    clf = KNN(n_neighbors=k, method='mean')
    clf.fit(df)
    scores_arr.append(clf.decision_scores_)
    scores_df = pd.DataFrame(scores_arr)
    scores_df['KNN Scores'] = scores_df.sum(axis=0)

scaler = MinMaxScaler()
for col_name in scores_df.columns:
    scores_df[col_name] = scaler.fit_transform(np.array(scores_df[col_name]).reshape(-1, 1))
scores_df['Avg Score'] = scores_df.sum(axis=1)


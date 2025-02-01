import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import RobustScaler,  OrdinalEncoder
from pyod.models.knn import KNN
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(0)

data = fetch_openml('abalone', version=1, parser='auto')
df = pd.DataFrame(data.data, columns=data.feature_names)
df = pd.DataFrame(OrdinalEncoder().fit_transform(df), columns=df.columns)
df = pd.DataFrame(RobustScaler().fit_transform(df), columns=df.columns)
orig_features = list(df.columns)

det = KNN()
det.fit(df)
df['KNN Score'] = det.decision_function(df)

combo_id = 0
scores_df = pd.DataFrame()
scores_2d_cols = []
for col_name_1_idx, col_name_1 in enumerate(orig_features):
    for col_name_2_idx, col_name_2 in enumerate(orig_features[col_name_1_idx+1:]):
        print(col_name_1, col_name_2)
        det = KNN()
        det.fit(df[[col_name_1, col_name_2]])
        scores_df[str(combo_id)] = det.decision_function(df[[col_name_1, col_name_2]])
        scores_2d_cols.append((col_name_1, col_name_2))
        combo_id += 1
scores_df['Final Score'] = scores_df.sum(axis=1)
scores_df = scores_df.sort_values(by='Final Score')
df['KNN 2D Scores'] = scores_df['Final Score']

sns.scatterplot(data=df, x='KNN Score', y='KNN 2D Scores')
plt.show()

df = df.sort_values(by='KNN 2D Scores')
max_row = df.iloc[-1:]
top_row_index = max_row.index[0]
top_feature_pairs_idxs = np.argsort(scores_df.loc[top_row_index])[::-1][1:5].values

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,8))
for pair_idx, pair_id in enumerate(top_feature_pairs_idxs):
    col_pairs = scores_2d_cols[pair_id]
    s = sns.scatterplot(data=df, x=col_pairs[0], y=col_pairs[1], alpha=0.3, ax=ax[pair_idx//2, pair_idx%2])
    s = sns.scatterplot(data=max_row, x=col_pairs[0], y=col_pairs[1], ax=ax[pair_idx//2, pair_idx%2], color='red', marker='*', s=200)

    s.set_title(f'{col_pairs[0]} \n AND \n{col_pairs[1]}')
    plt.tight_layout()
plt.show()
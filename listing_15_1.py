#listing_15_1
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import OrdinalEncoder
from pyod.models.iforest import IForest

np.random.seed(0)

X, y = fetch_kddcup99(subset="SA", percent10=True, random_state=42, return_X_y=True, as_frame=True)
y = (y != b"normal").astype(np.int32)
df = pd.DataFrame(OrdinalEncoder().fit_transform(X), columns=X.columns)
orig_feats = X.columns

clf_if = IForest()
clf_if.fit(df[orig_feats])
df['IF Scores'] = clf_if.decision_scores_

#listing_15_2
import matplotlib.pyplot as plt
import seaborn as sns
from pyod.models.ecod import ECOD

clf_ecod = ECOD()
clf_ecod.fit(df[orig_feats])
df['ECOD Scores'] = clf_ecod.decision_scores_

sns.scatterplot(data=df, x='IF Scores', y='ECOD Scores')
plt.show()

#listing_15_3
from counts_outlier_detector import CountsOutlierDetector
det = CountsOutlierDetector()
results = det.fit_predict(df[orig_feats])
det.explain_row(65050)

#listing_15_4
scores_df = pd.DataFrame()
col_map = {}
c = 0
for i_idx in range(len(orig_feats)):
    for j_idx in range(i_idx+1, len(orig_feats)):
        col_name_i = orig_feats[i_idx]
        col_name_j = orig_feats[j_idx]
        clf = IForest()
        clf.fit(df[[col_name_i, col_name_j]])
        scores_df = pd.concat([scores_df, pd.DataFrame({f'Scores_{c}': clf.decision_scores_})], axis=1)
        col_map[c] = [col_name_i, col_name_j]
        c += 1
def get_max_cols(x):
    return col_map[np.argmax(x)]

scores_cols = scores_df.columns
max_scores = scores_df.max(axis=1)
scores_df['Max Cols'] = scores_df.apply(get_max_cols, axis=1)
scores_df['Max Score'] = max_scores
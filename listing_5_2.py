import numpy as np
import pandas as pd

from sklearn.datasets import fetch_kddcup99
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns

X, y = fetch_kddcup99(subset="SA", percent10=True, random_state=42, return_X_y=True, as_frame=True)
y = (y != b"normal.").astype(np.int32)

cat_columns = ["protocol_type", "service", "flag"]
X_cat = pd.get_dummies(X[cat_columns])
col_names = [X.replace("'", "_") for x in X_cat.columns]
X_cat.columns = col_names
X_cat  = X_cat.reset_index()

numeric_cols = [x for x in X.columns if x not in cat_columns]
transformer = RobustScaler().fit(X[numeric_cols])
X_num = pd.DataFrame(transformer.transform(X[numeric_cols]))
X_num.columns = numeric_cols
X_num = X_num.reset_index()

X = pd.concat([X_cat, X_num], axis=1)

scores = []
for k in [5, 10, 20, 30, 40, 50, 60, 70, 80]:
    print(f"Executing loop for k={k}")
    det = LocalOutlierFactor(n_neighbors=k)
    pred = det.fit_predict(X)
    pred = 1.0 - det.negative_outlier_factor_
    r = roc_auc_score(y, pred)
    scores.append(r)

s = sns.lineplot(x=[5, 10, 20, 30, 40, 50, 60, 70, 80], y=scores)
s.set_ylabel("AUROC Score")
s.set_xlabel("k")
plt.title("AUROC given k")
plt.ylim(0.0 , 1.0)
plt.show()
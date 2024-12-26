import numpy as np
import pandas as pd
from sklearn.datasets import fetch_kddcup99
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(0)

X, y = fetch_kddcup99(subset="SA", percent10=True, random_state=42, return_X_y=True, as_frame=True)
y = (y != b"normal.").astype(np.int32)

enc = OrdinalEncoder()
X = enc.fit_transform(X)

det = IsolationForest()
det.fit(X)

pred = pd.Series(det.predict(X))
pred = pred.map({1: 0, -1: 1})
print(confusion_matrix(y, pred))

pred = det.score_samples(X)
min_score = pred.min()
max_score = pred.max()
pred = [( x - min_score ) / ( max_score - min_score ) for x in pred]
pred = [ 1.0 - x for x in pred]
roc_auc_score(y, pred)
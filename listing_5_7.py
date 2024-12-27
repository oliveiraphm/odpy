import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.datasets import fetch_kddcup99
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import roc_auc_score

np.random.seed(0)

X, y = fetch_kddcup99(subset="SA", percent10=True, random_state=42, return_X_y=True, as_frame=True)
y = (y != b"normal.").astype(np.int32)

enc = OrdinalEncoder() 
X = enc.fit_transform(X)

det = EllipticEnvelope()
pred = det.fit_predict(X)
r = roc_auc_score(y, pred)
print(r)
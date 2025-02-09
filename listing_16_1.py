import numpy as np
import pandas as pd
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import OrdinalEncoder

np.random.seed(0)
X, y = fetch_kddcup99(subset="SA", percent10=True, random_state=42,
                     return_X_y=True, as_frame=True)
y = (y != b"normal.").astype(np.int32)
enc = OrdinalEncoder() 
X = enc.fit_transform(X)

#listing_16_2.py
from pyod.models.auto_encoder import AutoEncoder

np.random.seed(0)
det = AutoEncoder(epochs=5, random_state=0)
det.fit(X)
pred = det.predict_proba(X)[:, 1]
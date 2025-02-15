import numpy as np
import pandas as pd
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import OrdinalEncoder
from deepod.models.tabular import GOAD


np.random.seed(0)
X, y = fetch_kddcup99(subset="SA", percent10=True, random_state=42,
                     return_X_y=True, as_frame=True)
y = (y != b"normal.").astype(np.int32)
enc = OrdinalEncoder() 
X = enc.fit_transform(X)

clf = GOAD()
clf.fit(X)
scores = clf.decision_function(X)
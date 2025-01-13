import numpy as np
import pandas as pd
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.gmm import GMM
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import OrdinalEncoder

np.random.seed(0)
df, _ = fetch_kddcup99(subset="SA", percent10=True, random_state=42,
                      return_X_y=True, as_frame=True)
enc = OrdinalEncoder() 
df = pd.DataFrame(enc.fit_transform(df), columns=df.columns)

outliers_df = df.copy() 
clf = IForest(contamination=0.15)
pred = clf.fit_predict(outliers_df)
outliers_df = outliers_df[pred == 1]

clf = LOF(contamination=0.15) 
pred = clf.fit_predict(outliers_df)
outliers_df = outliers_df[pred == 1]

clf = GMM(contamination=0.15) 
pred = clf.fit_predict(outliers_df)
outliers_df = outliers_df[pred == 1]

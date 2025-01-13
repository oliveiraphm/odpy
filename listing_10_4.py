import pandas as pd
import numpy as np
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import OrdinalEncoder

np.random.seed(0)

X, y = fetch_kddcup99(subset="SA", percent10=True, random_state=42,
                      return_X_y=True, as_frame=True)
y = (y != b"normal.").astype(np.int32)

enc = OrdinalEncoder() 
X = pd.DataFrame(enc.fit_transform(X), columns=X.columns)

from sklearn.preprocessing import RobustScaler
from pyod.models.cblof import CBLOF
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
np.random.seed(0)

training_set_sizes = [100, 250, 500, 1_000, 1_500, 2_000, 2_500, 3_000, 
                      4_000, 5_000, 10_000, 15_000, 20_000, 25_000]
number_unique_arr = []
corr_arr = []
for train_size in training_set_sizes: #A
  top_results = []
  scores_df = pd.DataFrame()
  for trial_number in range(10): #B
    det = CBLOF()
    det.fit(X.sample(n=train_size, random_state=trial_number))
    pred = det.decision_function(X)
    top_results.extend(np.argsort(pred)[:50]) #C
    scores_df[trial_number] = pred #D

  top_results = list(set(top_results)) #E
  number_unique_arr.append(len(top_results))
  
  for col_name in scores_df.columns: #F
    scaler = RobustScaler()
    scores_df[col_name] = scaler.fit_transform(
        np.array(scores_df[col_name]).reshape(-1, 1))
    scores_df[col_name] =\
      scores_df[col_name].apply(lambda x: x if x > 2.0 else 0.0)

  corr_arr.append(scores_df.corr().mean().mean())

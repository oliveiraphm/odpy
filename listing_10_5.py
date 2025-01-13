import numpy as np
import pandas as pd
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import OrdinalEncoder
from pyod.models.iforest import IForest
import concurrent
import warnings
warnings.filterwarnings('ignore')

np.random.seed(0)

X, y = fetch_kddcup99(subset="SA", percent10=True, random_state=42,
                      return_X_y=True, as_frame=True)

enc = OrdinalEncoder() 
X = pd.DataFrame(enc.fit_transform(X), columns=X.columns)
det = IForest() 
det.fit(X)  

num_splits = 5  

def evaluate_subset(df):  
  return det.decision_function(df)

process_arr = []
full_results = []
rows_per_subset = len(X) // num_splits
with concurrent.futures.ProcessPoolExecutor() as executor: 
    for dataset_idx in range(num_splits):
        print(f"Starting process for dataset: {dataset_idx}")
        start_row = dataset_idx * rows_per_subset
        end_row = (dataset_idx + 1) * rows_per_subset        
        print(start_row, end_row)
        f = executor.submit(evaluate_subset, X[start_row: end_row+1])
        process_arr.append(f)
    for f in process_arr: 
        full_results.extend(f.result())
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import roc_auc_score

X, y = fetch_kddcup99(
    subset="SA",  percent10=True, 
    random_state=42, return_X_y=True, as_frame=True)
y = (y != b"normal.").astype(np.int32)

cat_columns = ["protocol_type", "service", "flag"] 
X_cat = pd.get_dummies(X[cat_columns])
col_names = [x.replace("'", '_') for x in X_cat.columns]
X_cat.columns = col_names
X_cat = X_cat.reset_index()

num_cols = [x for x in X.columns if x not in cat_columns] 
transformer = RobustScaler().fit(X[num_cols])
X_num = pd.DataFrame(transformer.transform(X[num_cols]))
X_num.columns = num_cols
X_num = X_num.reset_index()

X = pd.concat([X_cat, X_num], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90, random_state=42) 
y_train = y_train[y_train == 0] 
X_train = X_train.loc[y_train.index] 

det = OneClassSVM() 
det.fit(X_train)

pred = det.score_samples(X_test) 
min_score = pred.min()
max_score = pred.max()
pred = [(x - min_score) / (max_score - min_score) for x in pred]
pred = [1.0 - x for x in pred]
print(roc_auc_score(y_test, pred))

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import RobustScaler

data = fetch_openml('baseball', version=1, parser='auto')
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Strikeouts'] = df['Strikeouts'].fillna(df['Strikeouts'].median())
df = pd.get_dummies(df)
scaler = RobustScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
orig_features = df.columns

#listing14_2

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from NearestSampleOutlierDetector import NearestSampleOutlierDetector
from RadiusOutlierDetector import RadiusOutlierDetector

scores_df = df.copy()

det_if = IsolationForest()
det_if.fit(df)
scores_df['IF Scores'] = (-1)*det_if.score_samples(df)

det_lof = LocalOutlierFactor()
det_lof.fit(df)
scores_df['LOF Scores'] = (-1)*det_lof.negative_outlier_factor_

det_ee = EllipticEnvelope()
det_ee.fit(df)
scores_df['EE Scores'] = (-1)*det_ee.score_samples(df)

det_ocsvm = OneClassSVM()
det_ocsvm.fit(df)
scores_df['OCSVM Scores'] = (-1)*det_ocsvm.score_samples(df)

det_ns = NearestSampleOutlierDetector()
det_ns.fit(df)
scores_df['Nearest Samples Scores'] = (-1)*det_ns.decision_function(df)

det_radius = RadiusOutlierDetector()
scores_df['Radius Scores'] = det_radius.fit_predict(df, radius=4.0)

#listing14_3

scores_cols = list(set(scores_df.columns) - set(df.columns))

for col_name in scores_cols:
    scores_df[col_name] = RobustScaler().fit_transform(scores_df[col_name].values.reshape(-1, 1)).reshape(1, -1)[0]

scores_df['Score'] = scores_df[scores_cols].sum(axis=1)

#listing14_4

import numpy as np
doped_df = df.sample(n=50)
for i in doped_df.index:
    row = doped_df.loc[i]
    col_name = np.random.choice(doped_df.columns)
    doped_df.loc[i, col_name] = doped_df.loc[i, col_name] * -0.9
df['Doped'] = 0
doped_df['Doped'] = 1
full_df = pd.concat([df, doped_df])
full_df = full_df.reset_index(drop=True)

from sklearn.metrics import roc_auc_score

ocsvm_scores_df = pd.DataFrame()
for i in range(50):
    sample_size = np.random.randint(300, 1300)
    sample_df = df.sample(n=sample_size)
    det_ocsvm = OneClassSVM(nu=0.2)
    det_ocsvm.fit(sample_df.values)
    ocsvm_scores_df[f'Scores {i}'] = (-1)*det_ocsvm.score_samples(full_df.values)

ocsvm_scores_df['Score'] = ocsvm_scores_df.sum(axis=1)
#print(roc_auc_score(full_df['Doped'], ocsvm_scores_df['Score']))

def get_top_scored(df, scores_col):
    cutoff = sorted(df[scores_col])[-10]
    return scores_df[scores_df[scores_col] >= cutoff].index.tolist()

scores_cols = set(scores_df.columns) - set(df.columns) 
top_rows = []
for scores_col in scores_cols:
    res = get_top_scored(scores_df, scores_col) 
    top_rows.extend(res)
top_rows = list(set(top_rows))

#listing14_7
import math
import itertools

def test_agreement(df, detector_cols, truth_col): 
    temp_df = scores_df.copy()
    temp_df['Test Score'] = temp_df[detector_cols].mean(axis=1)    
    temp_df['Test Score'] = temp_df['Test Score'].apply(
       lambda x: x if x > 2.0 else 0)
    temp_df[truth_col] = temp_df[truth_col].apply(
       lambda x: x if x > 2.0 else 0)
    corr = temp_df[['Test Score', truth_col]].corr().loc['Test Score', 
                                                         truth_col]
    if corr != corr:
        return 0.0
    return corr

scores_cols = list(set(scores_df.columns) - set(df.columns))
scores_cols.remove('Score')

for num_detectors in range(1, 6): 
    best_agreement = -1
    best_set_detectors = []
    num_combinations = math.comb(6, num_detectors)
    max_trials = 50 
    detector_cols_arr = itertools.combinations(scores_cols, num_detectors)
    if num_combinations <= max_trials: 
        for detector_cols in detector_cols_arr:
            res = test_agreement(df, list(detector_cols), 'Score')
            if res > best_agreement:
                best_agreement = res
                best_set_detectors = detector_cols    
    else: 
        for _ in range(max_trials): 
            detector_cols = np.random.choice(a=scores_cols,
                                             size=num_detectors)
            res = test_agreement(df, detector_cols, 'Score')
            if res > best_agreement:
                best_agreement = res
                best_set_detectors = detector_cols
    print((f"\nUsing {num_detectors} detectors, the best set of " 
           f"detectors found is: \n{best_set_detectors} with an "
           f"agreement score of {best_agreement}"))
    

#listing14_8

scores_cols = list(set(scores_df.columns) - set(df.columns))
scores_cols.remove('Score')

detectors_used = []
detectors_not_used = scores_cols.copy()

for num_detectors in range(6):
    best_next_detector = ""
    best_agreement = -1
    for detector in detectors_not_used:
        potential_set = detectors_used + [detector]
        res = test_agreement(df, potential_set, 'Score')
        if res > best_agreement:
            best_agreement = res
            best_next_detector = detector
    detectors_used.append(best_next_detector)
    detectors_not_used.remove(best_next_detector)
    print((f"Using {num_detectors}, the best set is:"
           f"{detectors_used} with an agreement score of "
           f"{best_agreement}"))
    

#listing14_9
weight = 0.7 
scores_cols = list(set(scores_df.columns) - set(df.columns))
scores_cols.remove('Score')

detectors_used = [] 
detectors_not_used = scores_cols.copy()

for num_detectors in range(6): 
    best_next_detector = ""
    best_agreement = -1
    for detector in detectors_not_used: 
        potential_set = detectors_used + [detector]
        agreement_target = test_agreement(
           scores_df, potential_set, 'Score')
        if len(detectors_used) > 0:
            agreement_ensemble = test_agreement(
               scores_df, detectors_used, detector)
        else:
            agreement_ensemble = 0.0 
        res = (weight*agreement_target) + (1.0-weight)*agreement_ensemble
        if res > best_agreement: 
            best_agreement = res
            best_next_detector = detector
    detectors_used.append(best_next_detector)
    detectors_not_used.remove(best_next_detector)
    print((f"Using {num_detectors}, the best set is: "
           f"{detectors_used} with an agreement score of "
           f"{best_agreement}"))

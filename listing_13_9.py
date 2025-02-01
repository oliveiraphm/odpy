import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.datasets import fetch_openml
from sklearn.ensemble import IsolationForest
import dice_ml

np.random.seed(0)

data = fetch_openml('abalone', version=1, parser='auto')
df = pd.DataFrame(data['data'], columns=data['feature_names'])
df = pd.DataFrame(OrdinalEncoder().fit_transform(df), columns=df.columns)
df = pd.DataFrame(RobustScaler().fit_transform(df), columns=df.columns)
orig_features = list(df.columns)

det = IsolationForest()
det.fit(df)
df['IF Score'] = det.score_samples(df)

class IF_wrapper:
    def __init__(self, det, cutoff):
        self.det = det
        self.cutoff = cutoff

    def predict_proba(self, x):
        return self.det.score_samples(x) < self.cutoff

if_wrapped = IF_wrapper(det, -0.6)
df['IF Outlier'] = df['IF Score'] < -0.6

d = dice_ml.Data(
    dataframe = df[orig_features + ['IF Outlier']],
    continuous_features = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight'],
    outcome_name = 'IF Outlier'
)

m = dice_ml.Model(model=if_wrapped, backend='sklearn')
exp = dice_ml.Dice(d, m, method='random')
query_instance = df.loc[2810:2810][orig_features]
dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=4, desired_class="opposite")
dice_exp.visualize_as_dataframe()
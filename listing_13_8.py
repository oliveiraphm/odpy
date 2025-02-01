from alibi.explainers import ALE, plot_ale
import warnings 
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.ensemble import IsolationForest
import warnings
from prism_rules import PrismRules

warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
np.random.seed(0)

data = fetch_openml('abalone', version=1, parser='auto') 
df = pd.DataFrame(data.data, columns=data.feature_names)
df = pd.DataFrame(OrdinalEncoder().fit_transform(df), columns=df.columns)
orig_features = list(df.columns)

det = IsolationForest(random_state=0) 
det.fit(df)

doped_df = df.copy() 
for i in doped_df.index:
  if doped_df.loc[i, 'Height'] > df['Height'].median():
    doped_df.loc[i, 'Height'] = doped_df.loc[i, 'Height'] / 10
  else:
    doped_df.loc[i, 'Height'] = doped_df.loc[i, 'Height'] * 2

df['Doped'] = 0 
doped_df['Doped'] = 1
test_df = pd.concat([df, doped_df])

test_df['IF Binary'] = det.predict(test_df[orig_features]) 

#prism = PrismRules(nbins=3) 
#_ = prism.get_prism_rules(test_df[orig_features + ['IF Binary']], 'IF Binary')

if_ale = ALE(det.score_samples, feature_names=orig_features, target_names=['IF Score'])
if_exp = if_ale.explain(df[orig_features].values)
plot_ale(if_exp, n_cols=4, fig_kw={'figwidth':14, 'figheight':7})
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy import stats

np.random.seed(0)

data = fetch_openml('baseball', version=1, parser='auto')
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Strikeouts'] = df['Strikeouts'].fillna(df['Strikeouts'].median())
df = pd.get_dummies(df)
df = pd.DataFrame(RobustScaler().fit_transform(df), columns=df.columns)

cat_features = [x for x in df.columns if df[x].nunique() <= 10]
num_features = [x for x in df.columns if x not in cat_features]

min_cols_per_modification = 1
max_cols_per_modification = 5
doped_df = df.copy().head(20)
for i in doped_df.index:
    num_cols_modified = -1
    while( num_cols_modified < min_cols_per_modification or num_cols_modified > max_cols_per_modification):
        num_cols_modified = int(abs(np.random.laplace(1.0, 10)))
    modified_cols = np.random.choice(df.columns, num_cols_modified, replace=False)

    for col_name in modified_cols:
        other_cols = df.columns.to_list()
        other_cols.remove(col_name)
        if col_name in num_features:
            regr = RandomForestRegressor()
            regr.fit(df[other_cols], df[col_name])
            pred = regr.predict(pd.DataFrame(doped_df.iloc[i][other_cols]).T)[0]
            pred_quantile = stats.percentileofscore(
               df[col_name], pred) // 25
            cur_val = doped_df.loc[i, col_name]
            cur_val_quantile = stats.percentileofscore(
               df[col_name], cur_val) // 25
            q1 = doped_df[col_name].quantile(0.25)
            q2 = doped_df[col_name].quantile(0.5)
            q3 = doped_df[col_name].quantile(0.75)
            quantiles_list = list(range(4))
            np.random.shuffle(quantiles_list)
            for q in quantiles_list: 
                if q != pred_quantile and q != cur_val_quantile:
                    break
            doped_df.loc[i, col_name] =\
                ((0.25) * q) + 0.125 + np.random.random()/20 

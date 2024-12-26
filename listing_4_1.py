import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler


data = fetch_openml('baseball', version=1, parser='auto')
df = pd.DataFrame(data.data, columns=data.feature_names)

print(df.shape)
print(df.head())
print(df.dtypes)

for col_name in df.columns:
    if(df[col_name].dtype in ['int64', 'float64']):
        sns.histplot(data=df[col_name])
        plt.show()

sns.countplot(data=df, y='Position', orient='h', color='blue')
plt.show()

for col_idx_1, col_name_1 in enumerate(df.columns):
    for col_idx_2 in range(col_idx_1+1, len(df.columns)):
        col_name_2 = df.columns[col_idx_2]
        if (df[col_name_1].dtype in ['int64', 'float64']) and \
           (df[col_name_2].dtype in ['int64', 'float64']):
            sns.scatterplot(data=df, x=col_name_1, y=col_name_2)
            plt.show()

for col_name_1 in df.columns:
    if(df[col_name_1].dtype in ['int64', 'float64']):
        sns.boxplot(data=df, x=col_name_1, y='Position', orient='h')
        plt.show()

df['Strikeouts'] = df['Strikeouts'].fillna(df['Strikeouts'].median())

limit_dict = {}

for col_name in df.columns:
    if (df[col_name].dtype in ['int64', 'float64']):
        q1 = df[col_name].quantile(0.25)
        q3 = df[col_name].quantile(0.75)
        iqr = q3 - q1
        limit = q3 + (3.5*iqr)
        limit_dict[col_name] = limit

cond = [True]*len(df)
for key, limit in limit_dict.items():
    cond = cond & (df[key] <= limit)

print(len(df))
clean_df = df[cond]
print(len(clean_df))

new_feat = 'Home runs per at_bats'
feat_1 = 'Home_runs'
feat_2 = 'At_bats'
df[new_feat] = df[feat_1] / df[feat_2]

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,3))
sns.scatterplot(x=df[feat_2], y=df[feat_1], ax=ax[0])
sns.histplot(df[new_feat], bins=100, ax=ax[1])
plt.show()

df = pd.get_dummies(df) 

scaler = RobustScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)


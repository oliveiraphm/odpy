import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import BallTree


data = fetch_openml('baseball', version=1, parser='auto')
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Strikeouts'] = df['Strikeouts'].fillna(df['Strikeouts'].median())
df = pd.get_dummies(df)
scaler = RobustScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df['DBSCAN Outliers'] = np.where( DBSCAN(eps=2.0).fit_predict(df) < 0, 1, 0)

df['DBSCAN Score'] = [0] * len(df)
for eps in np.arange(0.1, 5.0, 0.1):
    df['DBSCAN Score'] += np.where(DBSCAN(eps=eps).fit_predict(df) < 0, 1, 0)
sns.histplot(df['DBSCAN Score'])
plt.show()


tree = BallTree(df)
dist, ind = tree.query(df, k=4)
dist = [x.mean() for x in dist]
df['KNN Score'] = dist
df['KNN Outliers'] = df['KNN Score'] > 2.3

sns.histplot(df['KNN Score'])
plt.show() 
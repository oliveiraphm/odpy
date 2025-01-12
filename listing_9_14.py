from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.neighbors import BallTree
from listing_7_2 import create_four_clusters_test_data
import pandas as pd

df = create_four_clusters_test_data()

scaler = MinMaxScaler()
df_minmax = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

scaler = RobustScaler()
df_robust = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

balltree = BallTree(df_minmax)
dist, ind = balltree.query(df_minmax.loc[0:0], k=10)
print(ind)


balltree = BallTree(df_robust)
dist, ind = balltree.query(df_robust.loc[0:0], k=10)
print(ind)

df = pd.concat([df, pd.DataFrame({"A":[100], "B":[100]})])

scaler = MinMaxScaler()
df_minmax = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

scaler = RobustScaler()
df_robust = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

balltree = BallTree(df_minmax)
dist, ind- balltree.query(df_minmax.loc[0:0], k=10)
print(ind)

balltree = BallTree(df_robust)
dist, ind = balltree.query(df_robust.loc[0:0], k=10)
print(ind)

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.neighbors import BallTree

data = fetch_openml('segment', version=1, parser='auto')
data = pd.DataFrame(data.data, columns=data.feature_names)
X = data['hue-mean'].values.reshape(-1,1)

tree = BallTree(X, leaf_size=2)
dist, ind = tree.query(X, k=26)
max_dist_arr = pd.Series([max(x) for x in dist])
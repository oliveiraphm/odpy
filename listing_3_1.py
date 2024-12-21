import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(0)
n_per_cluster = 50 #500

data_a = pd.Series(np.random.laplace(size = n_per_cluster))
data_b = pd.Series(np.random.laplace(size = n_per_cluster))
df1 = pd.DataFrame({"A": data_a, "B": data_b})

data_a = pd.Series(np.random.normal(loc=5, size = n_per_cluster))
data_b = pd.Series(np.random.normal(loc=15, size = n_per_cluster))
df2 = pd.DataFrame({"A": data_a, "B": data_b})

df = pd.concat([df1, df2])

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,6))

s = sns.scatterplot(data=df, x="A", y="B", alpha=0.4, ax=ax[0][0])

tree = BallTree(df, leaf_size=100)
dist, _ = tree.query(df, k=3)
dist = [x.mean() for x in dist]
df['Score'] = dist
s = sns.scatterplot(data=df, x="A", y="B", s=50*df['Score'] ,alpha=0.4, ax=ax[0][1])

cutoff = sorted(df['Score'])[-6]
df['Outlier'] = df['Score'] > cutoff
sns.scatterplot(data=df[df['Outlier']==False], x="A", y="B", alpha=0.4, ax=ax[1][0])
sns.scatterplot(data=df[df['Outlier']==True], x="A", y="B", s=100, alpha=1.0, marker='*', ax=ax[1][0])

cutoff = sorted(df['Score'])[-16] 
df['Outlier'] = df['Score'] > cutoff
sns.scatterplot(data=df[df['Outlier']==False], x="A", y="B", alpha=0.4, ax=ax[1][1])
sns.scatterplot(data=df[df['Outlier']==True], x="A", y="B", s=100, alpha=1.0, marker='*', ax=ax[1][1])

plt.show()

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

tree = BallTree(df)
counts = tree.query_radius(df, 2.0, count_only=True)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8,4))
s = sns.histplot(counts, ax=ax[0])
s.set_title(f"Number of Points within Radius")

min_score = min(counts)
max_score = max(counts)
scores = [(max_score - x)/(max_score - min_score) for x in counts]
s = sns.histplot(scores, ax=ax[1])
s.set_title(f"Scores")

df['Score'] = scores
threshold = sorted(scores, reverse=True)[15]
df_flagged = df[df['Score'] >= threshold]
s = sns.scatterplot(data=df, x="A", y="B", ax=ax[2])
#s = sns.scatterplot(data=df_flagged, x="A", y="B", color='red', marker='*', s=200, ax=ax[2])

plt.tight_layout()
plt.show()
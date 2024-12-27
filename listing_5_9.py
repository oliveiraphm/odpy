from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from scipy import stats

np.random.seed(0)

vals, _ = make_blobs(n_samples=500, centers=3, n_features=2, cluster_std=1, random_state=42) 
df_inliers = pd.DataFrame(vals, columns=["A", "B"])

vals, _ = make_blobs(n_samples=15, centers=2, n_features=2, cluster_std=10, random_state=42)
df_outliers = pd.DataFrame(vals, columns=["A", "B"])

df = pd.concat([df_inliers, df_outliers])
X = df[['A', 'B']]

best_n_components = -1 
best_bic = np.inf
for n_components in range(1, 8):
    gmm = GaussianMixture(n_components=n_components, n_init=5, random_state=42)  
    gmm.fit(X)    
    bic = gmm.bic(X)
    if bic < best_bic:
        best_bic = bic
        best_n_components = n_components

gmm = GaussianMixture(n_components=best_n_components, n_init=5, random_state=42)  
gmm.fit(X)    

score = gmm.score_samples(X)
df['Score'] = score

pct_threshold = np.percentile(score, 3.0) 
df['Outlier'] = df['Score'] < pct_threshold

sns.scatterplot(data=df[df['Outlier'] == False], x='A', y='B', alpha=0.3) 
sns.scatterplot(data=df[df['Outlier'] == True], x='A', y='B', s=150, marker='*')
plt.show()

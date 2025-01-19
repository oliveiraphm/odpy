import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest

data = fetch_openml('baseball', version=1, parser='auto')
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Strikeouts'] = df['Strikeouts'].fillna(df['Strikeouts'].median())
df = pd.get_dummies(df)

np.random.seed(0)
clf_if = IsolationForest()
clf_if.fit(df)
pred = clf_if.decision_function(df)
trimmed_df = df.loc[np.argsort(pred)[50:]]

best_score = np.inf
best_n_clusters = -1
for n_clusters in range(2,10):
    gmm = GaussianMixture(n_components=n_clusters)
    gmm.fit(trimmed_df)
    score = gmm.bic(trimmed_df)
    if score < best_score:
        best_score = score
        best_n_clusters = n_clusters

gmm = GaussianMixture(n_components=best_n_clusters)
gmm.fit(trimmed_df)

samples = gmm.sample(n_samples=500)
synth_df = pd.DataFrame(samples[0], columns=df.columns)

sns.scatterplot(data=df, x='At_bats', y='RBIs', color='blue', alpha=0.1)
sns.scatterplot(data=synth_df, x='At_bats', y='RBIs', color='red', marker="*", s=200)
plt.show()

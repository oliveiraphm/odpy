import pandas as pd
import numpy as np
from pyod.models.knn import KNN
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

x_data = np.random.normal(loc=10, scale=1.0, size=1000)
y_data = np.random.normal(loc=10, scale=1.0, size=1000)
df1 = pd.DataFrame({'A': x_data, 'B': y_data})
df1['Ground Truth'] = 0

x_data = np.random.normal(loc=8, scale=1.0, size=5)
y_data = np.random.normal(loc=8, scale=1.0, size=5)
df2 = pd.DataFrame({'A': x_data, 'B': y_data})
df2['Ground Truth'] = 1

x_data = np.random.normal(loc=1, scale=3.0, size=1000)
y_data = np.random.normal(loc=1, scale=3.0, size=1000)
df3 = pd.DataFrame({'A': x_data, 'B': y_data})
df3['Ground Truth'] = 1

df = pd.concat([df1, df2, df3])
df = df.reset_index()
df = pd.DataFrame(RobustScaler().fit_transform(df), columns=df.columns)

fig, ax = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(10, 3))
sns.scatterplot(data=df[df['Ground Truth'] == False], x='A', y='B', alpha=0.1, ax=ax[0])

s = sns.scatterplot(data=df[df['Ground Truth'] == True], x='A', y='B', alpha=1.0, s=100, marker='*', ax=ax[0])
s.set_title('Ground Truth')

clf = KNN()
clf.fit(df[['A', 'B']])

df['KNN Binary Prediction'] = clf.predict(df[['A', 'B']])
sns.scatterplot(data=df[df['KNN Binary Prediction'] == False], x='A', y='B', alpha=0.1, s=100, marker='*', ax=ax[1])
s = sns.scatterplot(data=df[df['KNN Binary Prediction'] == True], x='A', y='B', alpha=1.0, s=100, marker='*', ax=ax[1])
s.set_title('KNN Binary Prediction')

df['KNN Decision Scores'] = clf.decision_scores_
s = sns.scatterplot(data=df, x='A', y='B', hue='KNN Decision Scores', size = 'KNN Decision Scores', ax=ax[2])
s.get_legend().remove()
s.set_title('Training Decision \nScores')

df['Knn Decision Function'] = clf.decision_function(df[['A', 'B']])
s = sns.scatterplot(data=df, x='A', y='B', hue='Knn Decision Function', size = 'Knn Decision Function', ax=ax[3])
s.get_legend().remove()
s.set_title('Decision Function \nScores')

plt.tight_layout()
plt.show()

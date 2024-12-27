import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import fetch_openml

data = fetch_openml(name='mfeat-karhunen', version=1, parser='auto')
df = pd.DataFrame(data.data, columns = data.feature_names)

cov = EllipticEnvelope(random_state=0).fit(df[['att6', 'att7']])
pred = cov.predict(df[['att6', 'att7']])

df['Elliptic Score'] = pred
df['Elliptic Score'] = df['Elliptic Score'].map({-1:1, 1:0})
sns.scatterplot(data=df[df['Elliptic Score'] == 0], x='att6', y='att7', alpha = 0.3)
sns.scatterplot(data=df[df['Elliptic Score'] == 1], x='att6', y='att7', alpha = 1.0, s=200, marker='*')
plt.show()

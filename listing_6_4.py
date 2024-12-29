import pandas as pd
from pyod.models.hbos import HBOS
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns

data = fetch_openml("speech", version=1, parser='auto')
df = pd.DataFrame(data.data, columns=data.feature_names)
print(df.head())

det = HBOS()
det.fit(df)
pred = det.decision_scores_

sns.histplot(pred)
plt.show()

df['HBOS Score'] = pred
df['Outlier'] = df['HBOS Score'] > df['HBOS Score'].quantile(0.995)

fig, ax = plt.subplots(nrows=20, ncols=20, sharey=True, figsize=(65,65))

for i in range(20):
    for j in range(20):
        col_name = f"V{(i*20) + j + 1}"
        sns.boxplot(data=df, x=col_name, orient='h', y='Outlier', ax=ax[i,j])
plt.show()

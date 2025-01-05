import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ecod import ECOD

data = fetch_openml('abalone', version=1)
df = pd.DataFrame(data.data, columns=data.feature_names)
df = pd.get_dummies(df)
df_orig = df.copy()
df = pd.DataFrame(RobustScaler().fit_transform(df), columns=df.columns)

clf = IForest()
clf.fit(df)
if_scores = clf.decision_scores_
top_if_scores = np.argsort(if_scores)[::-1][:10]
clean_df = df.loc[[x for x in df.index if x not in top_if_scores]].copy()

doped_df = df.copy()
for i in doped_df.index:
    col_name = np.random.choice(df.columns)
    med_val = clean_df[col_name].median()
    if doped_df.loc[i, col_name] > med_val:
        doped_df.loc[i, col_name] = clean_df[col_name].quantile(np.random.random()/2)
    else:
        doped_df.loc[i, col_name] = clean_df[col_name].quantile(0.5 + np.random.random()/2)    

def test_detector(clf, title, df, clean_df, doped_df, ax):
    clf.fit(clean_df)
    df = df.copy()
    doped_df = doped_df.copy()
    df["Scores"] = clf.decision_function(df)
    df["Source"] = 'Real'
    doped_df["Scores"] = clf.decision_function(doped_df)
    doped_df["Source"] = 'Doped'
    test_df = pd.concat([df, doped_df])
    sns.boxplot(data=test_df, orient='h', x="Scores", y="Source", ax=ax)
    ax.set_title(title)

fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(10, 3))
test_detector(IForest(), "IForest", df, clean_df, doped_df, ax[0])
test_detector(LOF(), "LOF", df, clean_df, doped_df, ax[1])
test_detector(ECOD(), "ECOD", df, clean_df, doped_df, ax[2])
plt.tight_layout()
plt.show()

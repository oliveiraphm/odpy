import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ecod import ECOD
from sklearn.metrics import roc_auc_score

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

#listing 8.9
test_df = pd.concat([df, doped_df])
y_true = [0] * len(df) + [1] * len(doped_df)

clf = IForest()
clf.fit(clean_df)
y_pred = clf.predict(test_df)
if_auroc = roc_auc_score(y_true, y_pred)

print(f"IForest AUROC: {if_auroc}")

clf = LOF()
clf.fit(clean_df)
y_pred = clf.decision_function(test_df)
loc_auroc = roc_auc_score(y_true, y_pred)

print(f"LOF AUROC: {loc_auroc}")

def test_training_size(n_rows):
    clf = LOF()
    clf.fit(clean_df.sample(n=nrows))
    y_pred = clf.decision_function(test_df)
    lof_auroc = roc_auc_score(y_true, y_pred)
    return lof_auroc


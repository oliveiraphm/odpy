import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import BallTree

from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.gmm import GMM
from pyod.models.kde import KDE
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD

from listing_7_5 import LDOFOutlierDetector

data = fetch_openml(name='abalone', version=1, parser='auto')
df = pd.DataFrame(data.data, columns=data.feature_names)
df = pd.get_dummies(df)
df = pd.DataFrame(RobustScaler().fit_transform(df), columns=df.columns)

def score_records():
    scores_df = df.copy()

    # IForest
    clf = IForest()
    clf.fit(df)
    scores_df['IF Scores'] = clf.decision_scores_

    # LOF
    clf = LOF()
    clf.fit(df)
    scores_df['LOF Scores'] = clf.decision_scores_

    # OCSVM
    clf = OCSVM()
    clf.fit(df)
    scores_df['OCSVM Scores'] = clf.decision_scores_

    # GMM
    clf = GMM()
    clf.fit(df)
    scores_df['GMM Scores'] = clf.decision_scores_

    # KDE
    clf = KDE()
    clf.fit(df)
    scores_df['KDE Scores'] = clf.decision_scores_

    # KNN
    clf = KNN()
    clf.fit(df)
    scores_df['KNN Scores'] = clf.decision_scores_

    # HBOS
    clf = HBOS()
    clf.fit(df)
    scores_df['HBOS Scores'] = clf.decision_scores_

    # ECOD
    clf = ECOD()
    clf.fit(df)
    scores_df['ECOD Scores'] = clf.decision_scores_

    # COPOD
    clf = COPOD()
    clf.fit(df)
    scores_df['COPOD Scores'] = clf.decision_scores_

     #LDOFOutlierDetector
    clf = LDOFOutlierDetector()
    scores_df['LDOF Scores'] = clf.fit_predict(df, k=5)

    return scores_df

scores_df = score_records()

fig, ax = plt.subplots(figsize=(10 ,10))
scores_cols = [x for x in scores_df.columns if 'Scores' in x]
m = sns.color_palette("Blues", as_cmap=True)
sns.heatmap(scores_df[scores_cols].corr(method='spearman'), annot=True, cmap=m)
#plt.show()

#listing 8.6
#top_scores_df = scores_df[scores_cols].copy()
#for col_name in top_scores_df.columns: 
#  top_scores_df[col_name] = top_scores_df[col_name].rank()
#  top_scores_df[col_name] = top_scores_df[col_name].apply(lambda x: x if x > (len(df) * 0.95) else 0.0)

#listing 8.7
top_scores_df = scores_df[scores_cols].copy()
for col_name in top_scores_df.columns:
  scaler = RobustScaler()
  top_scores_df[col_name] = scaler.fit_transform(
      np.array(top_scores_df[col_name]).reshape(-1, 1))
  top_scores_df[col_name] =\
   top_scores_df[col_name].apply(lambda x: x if x > 2.0 else 0.0)

fig, ax = plt.subplots(figsize=(10, 10))
m = sns.color_palette("Blues", as_cmap=True)
sns.heatmap(top_scores_df.corr(method='spearman'), cmap=m, annot=True)
plt.show()

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import IsolationForest
import shap

np.random.seed(0)

data = fetch_openml('abalone', version=1, parser='auto')
df = pd.DataFrame(data.data, columns=data.feature_names)
df = pd.get_dummies(df)
orig_features = list(df.columns)

det = IsolationForest(random_state=0)
det.fit(df)
df['IF Score'] = det.score_samples(df)

df = df.sort_values(by='IF Score')
explainer = shap.Explainer(det)
explanation = explainer(df[orig_features])

#shap.initjs()
#shap.plots.bar(explanation)

#shap.initjs()
#shap.plots.beeswarm(explanation)

shap.initjs()
shap.plots.bar(explanation[0])


#listing13_3
from pyod.models.iforest import IForest
import matplotlib.pyplot as plt
import seaborn as sns

pyod_clf = IForest()
pyod_clf.fit(df)

s_list = sorted(list(zip(pyod_clf.feature_importances_, df.columns)), reverse=True)
importance_score, col_names = zip(*s_list)
sns.barplot(orient='h', y=np.array(col_names), x=np.array(importance_score))
plt.show()
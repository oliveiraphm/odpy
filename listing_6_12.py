#NÃO FUNCIONA
import pandas as pd
from pyod.models.iforest import IForest
from pyod.models.pca import PCA
from pyod.models.knn import KNN
from pyod.models.copod import COPOD
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.suod import SUOD
from sklearn.datasets import fetch_openml

data = fetch_openml(name='speech', version=1, parser='auto')
df = pd.DataFrame(data['data'], columns=data['feature_names'])

detector_list = [IForest(), PCA(), KNN(n_neighbors=10), KNN(n_neighbors=20), KNN(n_neighbors=30), KNN(n_neighbors=40), COPOD(), ABOD(), CBLOF()]

clf = SUOD(base_estimators=detector_list, n_jobs=2, combination='average', verbose=False)

clf.fit(df)
scores = clf.decision_scores_
print(scores)
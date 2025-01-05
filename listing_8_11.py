import time
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.gmm import GMM
from pyod.models.kde import KDE
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.pca import PCA
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import RobustScaler


def time_detector(clf):
  start_time = time.process_time()
  clf.fit(test_df)
  iteration_fit_results_arr.append(time.process_time() - start_time)

  start_time = time.process_time()
  clf.decision_function(test_df)
  iteration_predict_results_arr.append(time.process_time() - start_time)

data = fetch_openml('abalone', version=1) 
df = pd.DataFrame(data.data, columns=data.feature_names)
df = pd.get_dummies(df)
df = pd.DataFrame(RobustScaler().fit_transform(df), columns=df.columns)

fit_results_arr = [] 
predict_results_arr = []

for multiplier in [1, 5, 10, 15, 20, 25]: 
  test_df = pd.concat([df]*multiplier) 
  iteration_fit_results_arr = [len(test_df)]
  iteration_predict_results_arr = [len(test_df)]

  time_detector(clf = IForest()) 
  time_detector(clf = LOF())
  time_detector(clf = OCSVM())
  time_detector(clf = GMM())
  time_detector(clf = KDE())
  time_detector(clf = KNN())
  time_detector(clf = HBOS())
  time_detector(clf = ECOD())
  time_detector(clf = COPOD())
  time_detector(clf = ABOD())
  time_detector(clf = CBLOF())
  time_detector(clf = PCA())

  fit_results_arr.append(iteration_fit_results_arr)
  predict_results_arr.append(iteration_predict_results_arr)

col_names = ['Number Rows', 'IF', 'LOF', 'OCSVM', 'GMM', 'KDE', 'KNN', 'HBOS', 'ECOD', 'COPOD', 'ABOD', 'CBLOF', 'PCA']
fit_results_df = pd.DataFrame(fit_results_arr, columns=col_names)
print(fit_results_df)

predict_results_df = pd.DataFrame(predict_results_arr, columns=col_names)
print(predict_results_df)

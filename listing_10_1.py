import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.hbos import HBOS
from pyod.models.gmm import GMM
from pyod.models.abod import ABOD
import time

np.random.seed(0)

num_rows = 100_000
num_cols = 10
data_corr = pd.DataFrame({0: np.random.random(num_rows)}) 

for i in range(1, num_cols):
  data_corr[i] = data_corr[i-1] + (np.random.random(num_rows) / 10.0)

copy_row = data_corr[0].argmax()
data_corr.loc[num_rows-1, 2] = data_corr.loc[copy_row, 2]
data_corr.loc[num_rows-1, 4] = data_corr.loc[copy_row, 4]
data_corr.loc[num_rows-1, 6] = data_corr.loc[copy_row, 6]
data_corr.loc[num_rows-1, 8] = data_corr.loc[copy_row, 8]

start_time = time.process_time() 
pca = PCA(n_components=num_cols)
pca.fit(data_corr)
data_corr_pca = pd.DataFrame(pca.transform(data_corr), 
                             columns=[x for x in range(num_cols)])
print("Time for PCA tranformation:", (time.process_time() - start_time))

np.random.seed(0) 

data_extreme = pd.DataFrame()
for i in range(num_cols):
  data_extreme[i] = np.random.random(num_rows)

copy_row = data_extreme[0].argmax()
data_extreme.loc[num_rows-1, 2] = data_extreme[2].max() * 1.5
data_extreme.loc[num_rows-1, 4] = data_extreme[4].max() * 1.5
data_extreme.loc[num_rows-1, 6] = data_extreme[6].max() * 1.5
data_extreme.loc[num_rows-1, 8] = data_extreme[8].max() * 1.5

start_time = time.process_time() 
pca = PCA(n_components=num_cols)
pca.fit(data_corr)
data_extreme_pca = pd.DataFrame(pca.transform(data_corr), 
                                columns=[x for x in range(num_cols)])
print("Time for PCA tranformation:", (time.process_time() - start_time))

def evaluate_detector(df, clf, model_type): 
  if "ECOD" in model_type:
    clf = ECOD()
  start_time = time.process_time()
  clf.fit(df)
  time_for_fit = (time.process_time() - start_time)

  start_time = time.process_time()
  pred = clf.decision_function(df)
  time_for_predict = (time.process_time() - start_time)

  scores_df[f'{model_type} Scores'] = pred
  scores_df[f'{model_type} Rank'] =\
    scores_df[f'{model_type} Scores'].rank(ascending=False)

  print(f"{model_type:<20} Fit Time:     {time_for_fit:.2f}")
  print(f"{model_type:<20} Predict Time: {time_for_predict:.2f}")  


def evaluate_dataset_variations(df, df_pca, clf, model_name): 
  evaluate_detector(df, clf, model_name)
  evaluate_detector(df_pca, clf, f'{model_name} (PCA)')
  evaluate_detector(df_pca[[0, 1, 2]], clf, f'{model_name} (PCA - 1st 3)')
  evaluate_detector(df_pca[[7, 8, 9]], clf, f'{model_name} (PCA - last 3)')

def evaluate_dataset(df, df_pca): 
  clf = IForest()
  evaluate_dataset_variations(df, df_pca, clf, 'IF')

  clf = LOF(novelty=True)
  evaluate_dataset_variations(df, df_pca, clf, 'LOF')

  clf = ECOD()
  evaluate_dataset_variations(df, df_pca, clf, 'ECOD')

  clf = HBOS()
  evaluate_dataset_variations(df, df_pca, clf, 'HBOS')

  clf = GMM()
  evaluate_dataset_variations(df, df_pca, clf, 'GMM')

  clf = ABOD()
  evaluate_dataset_variations(df, df_pca, clf, 'ABOD')


scores_df = data_corr.copy()
evaluate_dataset(data_corr, data_corr_pca)
rank_columns = [x for x in scores_df.columns 
                if type(x) == str and 'Rank' in x]
print(scores_df[rank_columns].tail())

scores_df = data_extreme.copy()
evaluate_dataset(data_extreme, data_extreme_pca)
rank_columns = [x for x in scores_df.columns 
                if type(x) == str and 'Rank' in x]
print(scores_df[rank_columns].tail())

from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns

data = fetch_openml('wall-robot-navigation', version=1, parser='auto')
data_df = pd.DataFrame(data.data, columns=data.feature_names)
X = data_df[['V6', 'V19']]

clustering = DBSCAN().fit_predict(X) 

data_df['DBScan Outliers'] = np.where(clustering < 0, 1, 0) 
sns.scatterplot(data=data_df[data_df['DBScan Outliers']==False], x="V6",
                y="V19", alpha=0.2, color='blue')
sns.scatterplot(data=data_df[data_df['DBScan Outliers']==True], x="V6", 
                y="V19", marker='*', s=200, color='red')
plt.show()

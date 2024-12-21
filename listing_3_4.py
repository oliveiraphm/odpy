import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import seaborn as sns

data = fetch_openml('wall-robot-navigation', version=1, parser='auto')
data_df = pd.DataFrame(data.data, columns=data.feature_names)
X = data_df[['V6', 'V19']]
kde = KernelDensity(kernel='gaussian').fit(X)
log_density = kde.score_samples(X)
data_df['KDE Scores'] = -log_density
cutoff = sorted(data_df['KDE Scores'])[-30]
data_df['Outlier'] = data_df['KDE Scores'] > cutoff

sns.scatterplot(data=data_df[data_df['Outlier']==False], x = "V6", y = "V19", alpha=0.1)
sns.scatterplot(data=data_df[data_df['Outlier']==True], x = "V6", y = "V19", s=200, alpha=1.0, marker="*")
plt.show()
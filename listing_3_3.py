import pandas as pd
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns

data = fetch_openml('wall-robot-navigation', version=1, parser='auto')
data_df = pd.DataFrame(data.data, columns=data.feature_names)
sns.kdeplot(data = data_df, x = "V6", y="V19", fill=True)
sns.scatterplot(data=data_df, x = "V6", y = "V19", alpha=0.2)
plt.show()
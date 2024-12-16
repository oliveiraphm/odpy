import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns

data = fetch_openml('SpeedDating', version=1, parser='auto')
data = pd.DataFrame(data.data, columns=data.feature_names)

col_name = 'age_o'
data[col_name] = data[col_name].fillna(data[col_name].median()).astype(np.int64)
vc = data[col_name].value_counts()
cumm_frac = [vc.values[::-1][:x+1].sum() / len(data) for x in range(len(vc))]
cumm_frac = np.array(cumm_frac)
num_rare_vals = np.where(cumm_frac < 0.005)[0].max()
cut_off = vc.values[::-1][num_rare_vals]
min_count = vc[cut_off]

plt.subplots(figsize=(10,2))
s = sns.barplot(x=vc.index, y=vc.values, order=vc.index, color='blue')
s.axvline(len(vc) - num_rare_vals - 0.5)
s.set_title(col_name)
plt.show()
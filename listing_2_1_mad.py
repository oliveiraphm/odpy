import pandas as pd
import numpy as np
import statistics
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

def calc_MAD(data):
    median = statistics.median(data)
    deviations = [abs(x - median) for x in data]
    median_deviation = statistics.median(deviations)
    mad_scores = [abs(x - median) / median_deviation for x in data]

    return mad_scores

data = fetch_openml("abalone", version=1, parser='auto')
data = pd.DataFrame(data.data, columns=data.feature_names)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,3))

pd.Series(data['Whole_weight']).hist(bins=50,ax=ax[0])
ax[0].set_title("Whole_weight")

mad_scores = calc_MAD(data['Whole_weight'])
pd.Series(mad_scores).hist(bins=50, ax=ax[1])
ax[1].set_title("Distribution of MAD Scores")

mad_scores = calc_MAD(np.concatenate([data['Whole_weight'], [4.0]]))
pd.Series(mad_scores).hist(bins=50, ax=ax[2])
ax[2].set_title("MAD Scores given an outlier")

plt.tight_layout()
plt.show()
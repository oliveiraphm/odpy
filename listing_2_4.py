import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

data = fetch_openml('segment', version=1, parser='auto')
data = pd.DataFrame(data.data, columns=data.feature_names)

X = data['hue-mean'].values.reshape(-1,1)
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
kde_scores = pd.Series(kde.score_samples(X))

q1 = kde_scores.quantile(0.25)
q3 = kde_scores.quantile(0.75)
iqr = q3 - q1
threshold = q1 - ( 2.2 * iqr )
rare_values = [data['hue-mean'][x] for x in range(len(data)) if kde_scores[x] < threshold]

fig, ax = plt.subplots()
plt.hist(data['hue-mean'], bins=200)
for rare_value in rare_values:
    ax.axvline(rare_value, color='red', linestyle='-.')

plt.show()

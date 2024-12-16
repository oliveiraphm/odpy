import pandas as pd
from sklearn.datasets import fetch_openml
import statistics
import matplotlib.pyplot as plt

data = fetch_openml('segment', version=1, parser='auto')
data = pd.DataFrame(data.data, columns=data.feature_names)

histogram = pd.cut(data['hue-mean'], bins=10, retbins=True)[0]
counts = histogram.value_counts().sort_index()

rare_ranges = []
for v in counts.index:
    count = counts[v]
    if count < 10:
        rare_ranges.append(str(v))

rare_values = []
for i in range(len(data)):
    if str(histogram[i]) in rare_ranges:
        rare_values.append(data['hue-mean'][i])

fig, ax = plt.subplots()
plt.hist(data['hue-mean'], bins=10, density=True)
for rare_value in rare_values:
    ax.axvline(rare_value, color='red', linestyle='-.')
plt.xticks([statistics.mean([x.left, x.right]) for x in counts.index])
ax.set_xticklabels(range(10))
plt.show()
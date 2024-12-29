from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns

data = load_breast_cancer()['data']
col = 3

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(11, 2))
sns.histplot(data[:, col], ax=ax[0], bins=5)
sns.histplot(data[:, col], ax=ax[1], bins=20)
sns.histplot(data[:, col], ax=ax[2], bins=200)
sns.ecdfplot(data[:, col], ax=ax[3])
plt.tight_layout()
plt.show()
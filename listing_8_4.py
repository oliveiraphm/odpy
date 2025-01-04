from pyod.models.knn import KNN 
import matplotlib.pyplot as plt
from listing_7_2 import create_four_clusters_test_data
import numpy as np

def plot_scatterplot(k, ax): 
  clf = KNN(n_neighbors=k, method='mean')
  clf.fit(df)

  Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  ax.contourf(xx, yy, Z, cmap='Blues')
  ax.scatter(data=df, x="A", y='B', color='black')
  ax.set_title(f"k={k}")

df = create_four_clusters_test_data() 
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))

xx, yy = np.meshgrid( 
    np.linspace(df['A'].min(), df['A'].max(), 50), 
    np.linspace(df['B'].min(), df['B'].max(), 50))

plot_scatterplot(3, ax[0]) 
plot_scatterplot(10, ax[1])
plot_scatterplot(300, ax[2])

plt.axis('tight')
plt.legend().remove()
plt.show()
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

x_data = np.random.random(100) 
y_data = np.random.random(100) / 10.0

data = pd.DataFrame({'A': x_data, 'B': x_data + y_data}) 
data= pd.concat([data, 
   pd.DataFrame([[1.8, 1.8], [0.5, 0.1]], columns=['A', 'B'])])

pca = PCA(n_components=2) 
pca.fit(data)
print(pca.explained_variance_ratio_)

new_data = pd.DataFrame(pca.transform(data), columns=['0', '1']) 
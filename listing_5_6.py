import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet

np.random.seed(0)
a_data = np.random.laplace(size=1000)
b_data = np.random.normal(size=1000) + a_data
c_data = np.random.exponential(size=1000) + a_data + b_data
X = pd.DataFrame({"A": a_data, "B": b_data, "C": c_data})

cov = MinCovDet(random_state=0).fit(X)
print(cov.covariance_)
print(cov.mahalanobis(X))
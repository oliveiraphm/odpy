import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


np.random.seed(0)
n_points = 2000
dates = np.array('2005-01-01', dtype=np.datetime64) + np.arange(n_points)
data = 12*np.sin(4 * np.pi * np.arange(n_points)/365) + \
       50.0*np.array(range(n_points))/n_points + \
       np.random.normal(12, 2, n_points)
df = pd.DataFrame({'Values': data}, index=dates)
result = seasonal_decompose(df, model='additive', period=365)
result.plot()
plt.show()
import pandas as pd
import numpy as np
from pyod.models.copod import COPOD

np.random.seed(42)

a_data = np.random.normal(loc=10, scale=1.0, size=1000)
b_data = np.random.normal(loc=10, scale=1.0, size=1000)
df = pd.DataFrame({"A": a_data, "B": b_data})
df.loc[999, 'A'] = 15

clf = COPOD()
clf.fit(df)
clf.explain_outlier(999)
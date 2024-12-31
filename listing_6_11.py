import pandas as pd
import numpy as np
from pyod.models.sod import SOD

np.random.seed(0)

d = np.random.randn(100, 35)
d = pd.DataFrame(d)
d[9] = d[9] + d[8]
d.loc[99, 8] = 3.5
d.loc[99, 9] = -3.8

clf = SOD(ref_set=3, contamination=0.01)
d['SOD Scores'] = clf.fit(d)
d['SOD Scores'] = clf.labels_

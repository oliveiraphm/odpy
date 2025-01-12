import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

vals = np.array(
    ['Sales']*1000 + ['Marketing']*5000 + ['Engineering']*100 +
    ['HR']*10 + ['Communications']*3
)
df = pd.DataFrame({"C1": vals})
vc = df["C1"].value_counts()
map = {x:y for x, y in zip(vc.index, vc.values)}
df['Ordinal C1'] = df['C1'].map(map)

clf = LocalOutlierFactor(contamination=0.01)
df['LOF Score'] = clf.fit_predict(df[['Ordinal C1']])

print(df)
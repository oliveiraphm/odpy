import pandas as pd
from pyod.models.pca import PCA
from pyod.models.ecod import ECOD
from sklearn.datasets import fetch_openml

data = fetch_openml("speech", version=1, parser='auto') 
df = pd.DataFrame(data.data, columns=data.feature_names)
scores_df = df.copy()

clf = ECOD(contamination=0.01) 
clf.fit(df)
scores_df['ECOD Scores'] = clf.predict(df)

clean_df = df[scores_df['ECOD Scores'] == 0] 

clf = PCA(contamination=0.02) 
clf.fit(clean_df)
pred = clf.predict(df)
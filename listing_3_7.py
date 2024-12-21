import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_text

np.random.seed(0)

data = load_breast_cancer()
data_df = pd.DataFrame(data.data, columns=data.feature_names)
data_df['Real'] = True

synth_df = pd.DataFrame()
for col_name in data_df.columns:
    mean = data_df[col_name].mean()
    stddev = data_df[col_name].std()
    synth_df[col_name] = np.random.normal(loc = mean, scale=stddev, size=len(data_df))
synth_df['Real'] = False

train_df = pd.concat([data_df, synth_df])

clf = DecisionTreeClassifier( max_depth=7, random_state=0 )
clf.fit(train_df.drop(columns=['Real']), train_df['Real'])
pred = clf.predict(train_df.drop(columns=['Real']))
print(confusion_matrix(train_df['Real'], pred))
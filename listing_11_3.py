import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

data = fetch_openml('baseball', version=1, parser='auto')
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Strikeouts'] = df['Strikeouts'].fillna(df['Strikeouts'].median())

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

cat_features = [x for x in df.columns if df[x].nunique() <=10]
num_features = [x for x in df.columns if x not in cat_features]
synth_data = [] 

feature_0 = df.columns[0] 
hist = np.histogram(df[feature_0], density=True)
bin_centers = [(x+y)/2 for x, y in zip(hist[1][:-1], hist[1][1:])]
p = [x/sum(hist[0]) for x in hist[0]]
vals = np.random.choice(bin_centers, p=p, size=len(df)).astype(int)
vals = [x + (((1.0 * np.random.random()) - 0.5) * df[feature_0].std()) 
        for x in vals]
synth_data.append(vals)
synth_cols = [feature_0]

for col_name in df.columns[1:]: 
    print(col_name)
    synth_df = pd.DataFrame(synth_data).T
    synth_df.columns = synth_cols
    
    if col_name in num_features: 
        regr = RandomForestRegressor()
        regr.fit(df[synth_cols], df[col_name]) 
        pred = regr.predict(synth_df[synth_cols]) 
        vals = [x + (((1.0 * np.random.random()) - 0.5) * pred.std()) 
                for x in pred] 
        synth_data.append(vals)  
        
    if col_name in cat_features:
        clf = RandomForestClassifier()
        clf.fit(df[synth_cols], df[col_name])
        synth_data.append(clf.predict(synth_df[synth_cols]))  
        
    synth_cols.append(col_name)
    
synth_df = pd.DataFrame(synth_data).T
synth_df.columns = synth_cols
        
def generate_dataset(df, max_cols_used, use_left):
    feature_0 = df.columns[0]
    hist = np.histogram(df[feature_0], density=True)
    bin_centers = [(x+y)/2 for x, y in zip(hist[1][:-1], hist[1][1:])]
    p = [x/sum(hist[0]) for x in hist[0]]
    vals = np.random.choice(bin_centers, p=p, size=len(df)).astype(int)
    vals = [x + (((1.0 * np.random.random()) - 0.5) * df[feature_0].std()) 
            for x in vals]
    synth_data = []
    synth_data.append(vals)
    synth_cols = [feature_0]
    for col_name in df.columns[1:]:
        print(col_name)
        synth_df = pd.DataFrame(synth_data).T
        synth_df.columns = synth_cols
        if use_left:
            use_synth_cols = synth_cols[:max_cols_used]
        else:
            use_synth_cols = synth_cols[-max_cols_used:]

        if col_name in num_features:
            regr = RandomForestRegressor()
            regr.fit(df[use_synth_cols], df[col_name])
            pred = regr.predict(synth_df[use_synth_cols])
            vals = [x + (((1.0 * np.random.random()) - 0.5) * pred.std())
                    for x in pred]
            synth_data.append(vals)
        
        if col_name in cat_features:
            clf = RandomForestClassifier()
            clf.fit(df[use_synth_cols], df[col_name])
            synth_data.append(clf.predict(synth_df[use_synth_cols]))
        
        synth_cols.append(col_name)
    
    synth_df = pd.DataFrame(synth_data).T
    synth_df.columns = synth_cols
    return synth_df

synth_df = generate_dataset(df, max_cols_used=2, use_left=False)

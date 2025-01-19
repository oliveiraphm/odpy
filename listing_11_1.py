import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml

data = fetch_openml('baseball', version=1, parser='auto') 
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Strikeouts'] = df['Strikeouts'].fillna(df['Strikeouts'].median())

cat_features = [x for x in df.columns if df[x].nunique() <=10] 
num_features = [x for x in df.columns if x not in cat_features]

synth_data = []
    
for num_feat in num_features: 
    hist = np.histogram(df[num_feat], density=True) 
    bin_centers = [(x+y)/2 for x, y in zip(hist[1][:-1], hist[1][1:])]
    p = [x/sum(hist[0]) for x in hist[0]]
    vals = np.random.choice(bin_centers, p=p, size=len(df)).astype(int)
    vals = [x + (((1.0 * np.random.random()) - 0.5) * df[num_feat].std()) 
            for x in vals]
    synth_data.append(vals)

for cat_feat in cat_features: 
    vc = df[cat_feat].value_counts(normalize=True)
    vals = np.random.choice(list(vc.index), p=list(vc.values), 
                            size=len(df)) 
    synth_data.append(vals)
    
synth_df = pd.DataFrame(synth_data).T
synth_df.columns = num_features + cat_features

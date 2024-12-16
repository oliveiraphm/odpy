import pandas as pd
import numpy as np
import statistics
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_openml

def calc_MAD(data):
    median = statistics.median(data)
    deviations = [abs(x - median) for x in data]
    median_deviation = statistics.median(deviations)
    if median_deviation == 0:  
        mean_deviation = statistics.mean(deviations)
        return [abs(x - median) / mean_deviation for x in data] 
    return [abs(x - median) / median_deviation for x in data] 

data = fetch_openml('segment', version=1, parser='auto') 
df = pd.DataFrame(data.data, columns=data.feature_names)

total_mad_scores = [0]*len(df)
for col_name in df.columns: 
    if df[col_name].nunique() == 1:
        continue
    mad_scores = calc_MAD(df[col_name])
    mad_scores = np.array(mad_scores).reshape(-1, 1)
    transformer = MinMaxScaler().fit(mad_scores) 
    col_mad_scores = transformer.transform(mad_scores).reshape(1, -1)[0]
    total_mad_scores += col_mad_scores 
print(total_mad_scores)

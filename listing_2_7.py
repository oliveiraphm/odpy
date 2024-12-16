from sklearn.datasets import fetch_openml
import pandas as pd

data = fetch_openml('eucalyptus', version=1, parser='auto')
data = pd.DataFrame(data.data, columns=data.feature_names) 
print(data[['Abbrev', 'Locality', 'Sp']].value_counts())
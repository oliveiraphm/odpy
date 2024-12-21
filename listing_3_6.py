from mlxtend.frequent_patterns import apriori
import pandas as pd
from sklearn.datasets import fetch_openml
import warnings

warnings.filterwarnings(action='ignore', category=DeprecationWarning)

data = fetch_openml('SpeedDating', version=1, parser='auto')
data_df = pd.DataFrame(data.data, columns=data.feature_names)

data_df = data_df[['d_pref_o_attractive', 'd_pref_o_sincere',
              'd_pref_o_intelligence', 'd_pref_o_funny',
              'd_pref_o_ambitious', 'd_pref_o_shared_interests']]
data_df = pd.get_dummies(data_df)
data_df = data_df.replace(1, True)
data_df = data_df.replace(0, False)

frequent_itemsets = apriori(data_df, min_support=0.3, use_colnames=True)
data_df['FPOF_Score'] = 0

for fis_idx in frequent_itemsets.index:
    fis = frequent_itemsets.loc[fis_idx, 'itemsets']
    support = frequent_itemsets.loc[fis_idx, 'support']
    col_list = (list(fis))
    cond = True
    for col_name in col_list:
        cond = cond & (data_df[col_name])

    data_df.loc[data_df[cond].index, 'FPOF_Score'] += support

min_score = data_df['FPOF_Score'].min()
max_score = data_df['FPOF_Score'].max()
data_df['FPOF_Score'] = [(max_score - x)/(max_score - min_score) for x in data_df['FPOF_Score']]

print(data_df)
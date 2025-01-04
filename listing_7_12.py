from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

data = fetch_openml('SpeedDating', version=1, parser='auto')
data_df = pd.DataFrame(data.data, columns=data.feature_names)
data_df = data_df[['d_pref_o_attractive', 'd_pref_o_sincere',  
                   'd_pref_o_intelligence', 'd_pref_o_funny', 
                   'd_pref_o_ambitious', 'd_pref_o_shared_interests']]
data_df = pd.get_dummies(data_df)
frequent_itemsets = apriori(data_df, min_support=0.3, use_colnames=True)
assoc_rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.7,num_itemsets=2)

data_df['Assoc Rules Score'] = 0
for assoc_rule_idx in assoc_rules.index:
    antecedents = assoc_rules.loc[assoc_rule_idx, 'antecedents']
    consequent = assoc_rules.loc[assoc_rule_idx, 'consequents']
    support = assoc_rules.loc[assoc_rule_idx, 'support']
    cond = True
    col_list = (list(antecedents))
    for col_name in col_list:
        cond = cond & (data_df[col_name])
    fis_true_list = data_df[cond].index
    col_list = (list(consequent))
    for col_name in col_list:
        cond = cond & (data_df[col_name])
    assoc_rule_true_list = data_df[cond].index
    rule_exceptions = list(set(fis_true_list) - set(assoc_rule_true_list))
    data_df.loc[rule_exceptions, 'Assoc Rules Score'] += support

print(data_df)
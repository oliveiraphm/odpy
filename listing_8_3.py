import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.Dataframe(columns = ['A'])
scores_df = pd.Dataframe(columns = ['Outlier'])
s = sns.histplot(df['A'])
s = sns.histplot(df['A'])
sub_df = scores_df[scores_df['Outlier'] == True]
for i in sub_df.index:
  a_val = df.loc[i, 'A']                             
  s.axvline(a_val, color='red', linestyle='-.')
plt.show()

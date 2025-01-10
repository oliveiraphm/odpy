import pandas as pd
import numpy as np

df = pd.DataFrame({"A": [1, 2, np.NaN, 4, 5, None],
                   "B": ['A', 'B', None, "B", "B", "F"]})

df_filled = df.copy()
df_filled['A Null'] = df['A'].isna()
df_filled['B Null'] = df['B'].isna()
df_filled['Num Null'] = df.isna().sum(axis=1)
df_filled['A'] = df['A'].fillna(df['A'].median())
df_filled['B'] = df['B'].fillna(df['B'].mode()[0])
print(df_filled)
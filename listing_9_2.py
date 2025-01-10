import pandas as pd
import numpy as np
df = pd.DataFrame({"A": [1, 2, np.NaN, 4, 5, None], 
                   "B": ['A', 'B', None, np.NaN, "E", "F"]})
df_filled = df.isna()
print(df_filled)
import pandas as pd
import numpy as np
import dask.dataframe as dd
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.datasets import fetch_kddcup99

X, y = fetch_kddcup99(subset="SA", percent10=True, random_state=42, return_X_y=True, as_frame=True)

enc = OrdinalEncoder()
df = pd.DataFrame(enc.fit_transform(X), columns=X.columns)

dask_df = dd.from_pandas(df, npartitions=2)
transformer = RobustScaler().fit(dask_df)
vals = transformer.transform(dask_df)
dask_df = dd.from_array(vals, columns=df.columns)

dask_df['Max Value'] = dask_df.max(axis=1)
dask_df.compute().sort_values('Max Value')
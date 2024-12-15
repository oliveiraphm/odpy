import pandas as pd
import numpy as np

data = pd.Series(np.random.normal(size=10000))
q1 = data.quantile(0.25)
q3 = data.quantile(0.75)
iqr = q3 - q1
iqr_lower_limit = q1 - (2.2 * iqr)
iqr_upper_limit = q3 + (2.2 * iqr)

print(f"lower limit: {iqr_lower_limit} \nupper limit: {iqr_upper_limit}")
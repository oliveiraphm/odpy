import pandas as pd
import numpy as np
from pyod.models.ecod import ECOD
from sklearn.datasets import fetch_openml

data = fetch_openml("speech", version=1, parser='auto')
df = pd.DataFrame(data.data, columns=data.feature_names)

det = ECOD()
det.fit(df)
pred = det.decision_scores_

det.explain_outlier(np.argmax(pred))


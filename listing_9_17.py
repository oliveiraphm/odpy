import math
import pandas as pd
import numpy as np


def eskin(df):
    n_rows = len(df)
    n_cols = len(df.columns)
    num_cat = [df[col_name].nunique() for col_name in df.columns]
    n_squared = [math.pow(x, 2) for x in num_cat]
    mismatch_score = [x/(x+2) for x in n_squared]
    eskin_scores = np.zeros((n_rows, n_rows))

    for i in range(n_rows-1):
        for j in range(1+i, n_rows):
            pair_distance = [1]*n_cols
            for k in range(n_cols):
                if df.loc[i, df.columns[k]] != df.loc[j, df.columns[k]]:
                    pair_distance[k] = mismatch_score[k]
            eskin_scores[i][j] = (n_cols/sum(pair_distance)) - 1
            eskin_scores[j][i] = eskin_scores[i][j]
    return eskin_scores

df = pd.DataFrame({"Department": [0, 1, 2, 0, 0],
                   "Account": [0, 1, 2, 1, 2],
                   "Date of Expence": [2, 2, 3, 4, 4],
                   "Date Submitted": [3, 5, 5, 7, 7]})

print(eskin(df))
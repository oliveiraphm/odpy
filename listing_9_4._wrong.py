from datetime import datetime
import pandas as pd
import numpy as np

df = pd.DataFrame({'Date of Expense': ['2021-01-01', '2021-01-02', '2021-01-03'],
                   'Date Submitted': ['2021-01-02', '2021-01-03', '2021-01-04']})

df['DOW 1'] = df['Date of Expense'].dt.dayofweek 
df['DOM 1'] = df['Date of Expense'].dt.day
df['Beg. of Month'] = df['DOM 1'].apply(lambda x: x < 15, 1, 0)
df['Gap'] = (df['Date Submitted'] - df['Date of Expense']).dt.days
df = df.drop(columns=['Date of Expense', 'Date Submitted']) 

print(df)
from datetime import datetime
import pandas as pd
import numpy as np

# Create the DataFrame
df = pd.DataFrame({'Date of Expense': ['2021-01-01', '2021-01-02', '2021-01-03'],
                   'Date Submitted': ['2021-01-02', '2021-01-03', '2021-01-04']})

# Convert to datetime
df['Date of Expense'] = pd.to_datetime(df['Date of Expense'])
df['Date Submitted'] = pd.to_datetime(df['Date Submitted'])

# Extract day of week and day of month
df['DOW 1'] = df['Date of Expense'].dt.dayofweek
df['DOM 1'] = df['Date of Expense'].dt.day

# Apply lambda function for beginning of month
df['Beg. of Month'] = df['DOM 1'].apply(lambda x: 1 if x < 15 else 0)

# Calculate gap in days
df['Gap'] = (df['Date Submitted'] - df['Date of Expense']).dt.days

# Drop original columns
df = df.drop(columns=['Date of Expense', 'Date Submitted'])

print(df)
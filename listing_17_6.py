import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

np.random.seed(42)
n_points = 2000
dates = np.array('2005-01-01', dtype=np.datetime64) + np.arange(n_points)
d_df = pd.DataFrame(dates)

base_value = 100
values = []
prev_value = base_value
for i in range(n_points):
    dow = d_df.loc[i, 0].dayofweek
    r = (np.random.rand() *  2) - 1
    v = (0.8) * prev_value + (0.02) * (r * base_value) + (0.8) *(dow+1)
    values.append(v)
df = pd.DataFrame({'Values': values}, index=dates)
df.iloc[1980]['Values'] = df.iloc[1980]['Values'] * 1.1 #listing_17_9
#fig, ax = plt.subplots(figsize=(15,2))
#sns.lineplot(df['Values'], color='blue')
#plt.show()


#listing_17_7
from darts import TimeSeries
from darts.models import ARIMA

df['timestamp'] = df.index
series = TimeSeries.from_dataframe(df, 'timestamp', 'Values')

model = ARIMA(p=7, d=2, q=2)
model.fit(series[:-60])
pred = model.predict(60)

#pred.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
#plt.legend()
#plt.show()

#listing_17_8
from sklearn.ensemble import RandomForestRegressor

df['Lag 1'] = df['Values'].shift(1)
df['Lag 7'] = df['Values'].shift(7)
df['Lag 14'] = df['Values'].shift(14)
df['Lag 28'] = df['Values'].shift(28)
df['Lag 30'] = df['Values'].shift(30)
df['Lag 360'] = df['Values'].shift(360)
df['Lag 364'] = df['Values'].shift(364)

x_train = df.drop(columns=['timestamp', 'Values'])[:1000]
y_train = df['Values'][:1000]
x_train = x_train.dropna()
y_train = y_train.loc[x_train.index]
x_test = df.drop(columns=['timestamp', 'Values'])[1000:]
y_test = df['Values'][1000:]

regr = RandomForestRegressor()
regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)

#listing_17_9
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true-y_pred)/y_true) * 100.0)

mape(y_test, y_pred)


#listing_17_10
avg_frac_dict = {x : [] for x in range(7)}
for i in range(0, len(df), 7):
    week_df = df.iloc[i: i+7]
    avg_val = week_df['Values'].mean()
    for j in range(len(week_df)):
        avg_frac_dict[j].append(week_df.iloc[j]['Values']/avg_val)
avg_frac_dict = {x : np.mean(avg_frac_dict[x]) for x in avg_frac_dict.keys()}

#listing_17_11

import statistics
df['Lag -1'] = df['Values'].shift(-1)
df['Bigger'] = df['Values'] > df['Lag 1']
df['Peak'] = (df['Values'] > df['Lag 1']) & (df['Values'] > df['Lag -1'])
df['YM'] = df.index.strftime("%Y%m")
df.groupby(['YM'])['Values'].std()
df.groupby(['YM'])['Bigger'].mean()
df.groupby(['YM'])['Peak'].mean()

#listing_17_12
weekly_arr = []
for i in range(0, len(df), 7): 
   week_df = df.iloc[i: i+7]
   weekly_arr.append([i] + week_df['Values'].values.tolist())
weekly_df = pd.DataFrame(
    weekly_arr, 
    columns=['Week', 'Mon', 'Tues', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun'])
weekly_df = weekly_df.dropna()
weekly_df
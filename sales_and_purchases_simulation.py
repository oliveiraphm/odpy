import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import statistics
from scipy.spatial import distance


class Purchacer:
    
    def __init__(self, staff_id):
        self.staff_id = staff_id

    def consider_purchase(self):
        global inventory
        global transactions
        if(current_datetime.hour < 9) or (current_datetime.hour > 17):
            return
        if(np.random.rand() > 0.01):
            product_id = np.random.choice(range(num_items_in_inventory))
            if inventory[product_id] > 1000:
                return
            supplier_id = np.random.choice(range(num_suppliers))
            unit_cost = abs(np.random.normal()) * 20.0
            num_purchased = np.random.randint(10, 200)
            total_cost = unit_cost * num_purchased
            inventory[product_id] += num_purchased
            transactions.append(['Purchase', self.staff_id, supplier_id, product_id, 
                                 current_datetime.strftime("%Y-%m-%d %H:%M"),
                                 num_purchased, unit_cost, total_cost,
                                 inventory[product_id]]
                                )

class RoguePurchaser:
    def __init__(self, staff_id):
        self.staff_id = staff_id
        self.extra_purchase_months = []

    def consider_purchase(self):
        global inventory
        global transactions
        if(current_datetime.hour < 9) or (current_datetime.hour > 17):
            return
        
        if (current_datetime.day == 28) and (current_datetime.month not in self.extra_purchase_months): 
            product_id = 5
            supplier_id = 10
            unit_cost = abs(np.random.normal()) * 60.0  # 40.0
            #trend_factor = np.log2(current_datetime.month+1) + 1
            trend_factor = current_datetime.month + 1
            num_purchased = int(np.random.randint(50, 60) * trend_factor)  
            #print(current_datetime.month, trend_factor, num_purchased)
            total_cost = num_purchased * unit_cost
            inventory[product_id] += num_purchased        
            transactions.append(['Purchase', self.staff_id, supplier_id, product_id, 
                                 current_datetime.strftime("%Y-%m-%d %H:%M"),
                                 num_purchased, unit_cost, total_cost, 
                                 inventory[product_id]])            
            self.extra_purchase_months.append(current_datetime.month)
        
        elif np.random.random() > 0.01:
            product_id = np.random.choice(range(num_items_in_inventory))
            if product_id == 0: 
                return            
            if inventory[product_id] > 1000:
                return        
            supplier_id = np.random.choice(range(num_suppliers))
            unit_cost = abs(np.random.normal()) * 20.0
            num_purchased = np.random.randint(10, 200)
            total_cost = num_purchased * unit_cost
            inventory[product_id] += num_purchased        
            transactions.append(['Purchase', self.staff_id, supplier_id, product_id, 
                                 current_datetime.strftime("%Y-%m-%d %H:%M"),
                                 num_purchased, unit_cost, total_cost, 
                                 inventory[product_id]])
            
class Sales: 
    def consider_sale(self):
        global inventory
        if (current_datetime.hour < 9) or (current_datetime.hour > 17):
            return
        if np.random.random() > 0.8:
            num_sold = np.random.randint(1, 50)
            product_id = np.random.choice(range(num_items_in_inventory))
            if num_sold < inventory[product_id]:
                inventory[product_id] -= num_sold   
                transactions.append(['Sale', 100, -1, product_id, 
                                     current_datetime.strftime("%Y-%m-%d %H:%M"), 
                                     num_sold, -1, -1, inventory[product_id]])
    
    
np.random.seed(0)
    
current_datetime = datetime(2022, 12, 15) 
end_date = datetime(2023, 12, 31)
delta = timedelta(minutes=1)
num_items_in_inventory = 20
inventory = [0]*num_items_in_inventory 
num_purchasers = 10
num_suppliers = 20
list_purchasers = list(range(num_purchasers))

purchasers_arr = [Purchacer(x) for x in range(num_purchasers)]
rogue_purchaser = RoguePurchaser(num_purchasers)
seller = Sales()
transactions = []

while current_datetime <= end_date:       
    np.random.shuffle(list_purchasers)
    for p in list_purchasers:
        purchasers_arr[p].consider_purchase()
    rogue_purchaser.consider_purchase()
    seller.consider_sale()
    
    current_datetime += delta
    
transactions_df = pd.DataFrame(transactions, columns=[
    'Type', 'Staff ID', 'Supplier ID', 'Product ID', 
    'Datetime', 'Count', 'Unit Cost', 'Total Cost', 'Inventory'])
transactions_df = transactions_df[pd.to_datetime(transactions_df['Datetime']) >= datetime(2023, 1, 1)]

purchases_df = transactions_df[transactions_df['Type']=='Purchase'].copy() 
purchases_df = purchases_df.drop(columns=['Type'])

purchases_df['Date'] = pd.to_datetime(purchases_df['Datetime']).dt.date
purchases_df['Year'] = pd.to_datetime(purchases_df['Datetime']).dt.year
purchases_df['Month'] = pd.to_datetime(purchases_df['Datetime']).dt.month
purchases_df['Day'] = pd.to_datetime(purchases_df['Datetime']).dt.day
purchases_df['Hour'] = pd.to_datetime(purchases_df['Datetime']).dt.hour
purchases_df['Minute'] = pd.to_datetime(purchases_df['Datetime']).dt.minute
purchases_df = purchases_df.drop(columns=['Datetime'])
purchases_df = purchases_df.reset_index(drop=True)
purchases_df.insert(0, 'Purchase ID', purchases_df.index)

duplicates_test_df = (pd.DataFrame(
    purchases_df.groupby(
        ['Date', 'Hour', 'Staff ID', 'Product ID']
    ).size(),
    columns=['Count']
)).reset_index()
duplicates_test_df['Count'].value_counts()

duplicates_test_df = (pd.DataFrame(
    purchases_df.groupby(
        ['Date', 'Hour', 'Staff ID', 'Supplier ID', 'Product ID']
    ).size(),
    columns=['Count']
)).reset_index()
duplicates_test_df['Count'].value_counts()

purchases_df = purchases_df.sort_values(['Date'])
purchases_df['Date'].diff().dt.days.value_counts()

counts = pd.crosstab(index=purchases_df['Staff ID'], columns=purchases_df['Product ID'], values=purchases_df['Count'], aggfunc='sum')
counts = counts.fillna(0)
#sns.histplot(counts.values.flatten(), bins=30)
#plt.show()

unusual_products_df = purchases_df.groupby(['Product ID']).agg({
    'Count': ['mean', 'sum']
    , 'Unit Cost': ['mean', 'sum']
    , 'Total Cost': ['mean', 'sum']
    , 'Hour': ['mean', 'min', 'max']
    , 'Purchase ID': ['count']
})

#print(unusual_products_df)

agg_df = purchases_df.groupby(['Staff ID', 'Product ID']).agg({
    'Count': ['mean', 'sum']
    , 'Unit Cost': ['mean', 'sum']
    , 'Total Cost': ['mean', 'sum']
    , 'Hour': ['mean', 'min', 'max']
    , 'Purchase ID': ['count']
})

#print(agg_df)

det = IsolationForest()
det.fit(agg_df)
agg_df['IF Scores'] = det.decision_function(agg_df)
agg_df.sort_values(['IF Scores'])

agg_df_day = purchases_df.groupby(['Date']).agg({
    'Count': ['mean', 'sum']
    , 'Unit Cost': ['mean', 'sum']
    , 'Total Cost': ['mean', 'sum']
    , 'Purchase ID': ['count']
})

#print(agg_df_day)

day_28_df = purchases_df[purchases_df['Day'] == 28]

def p95(x):
    return x.quantile(0.95)

purchases_df.groupby(['Staff ID', 'Day']).agg({
    'Count': ['mean', 'sum'], 
    'Total Cost': ['mean', 'sum', p95],  
    'Purchase ID': ['count']})
#sns.histplot(purchases_df.groupby(['Staff ID', 'Day'])['Total Cost'].sum())
#plt.show()

sub_df = purchases_df[(purchases_df['Staff ID'] == 10) & (purchases_df['Day'] == 28) & (purchases_df['Product ID'] == 5)]
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,3))
s = sns.lineplot(data=sub_df, x='Date', y='Count', ax=ax[0])
s.set_xticklabels([])
s = sns.lineplot(data=sub_df, x='Date', y='Unit Cost', ax=ax[1])
s.set_xticklabels([])
s = sns.lineplot(data=sub_df, x='Date', y='Total Cost', ax=ax[2])
s.set_xticklabels([])

plt.tight_layout()
plt.show()

def trend_func(x):
    return statistics.mean(x[-3:]) / statistics.mean(x[:3])

purchases_df = purchases_df.sort_values(['Staff ID', 'Product ID', 'Date'])
staff_product_day_df = purchases_df.groupby(
    ['Staff ID', 'Product ID', 'Day'], as_index=False).agg(
    {'Purchase ID': 'count',
     'Count': ['first','last', trend_func],
     'Unit Cost': ['first','last', trend_func],
     'Total Cost': ['first','last', trend_func]
    })
staff_product_day_df['Count Diff Last to First'] = staff_product_day_df[('Count', 'last')] - staff_product_day_df[('Count', 'first')]
#staff_product_day_df

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(23,3))
sns.countplot(data=purchases_df, x = 'Supplier ID', ax=ax[0][0])
sns.countplot(data=purchases_df[purchases_df['Staff ID']==0], x = 'Supplier ID', ax=ax[0][1])
sns.countplot(data=purchases_df[purchases_df['Staff ID']==1], x = 'Supplier ID', ax=ax[1][0])
sns.countplot(data=purchases_df[purchases_df['Staff ID']==10], x = 'Supplier ID', ax=ax[1][1])
plt.show()


base_distribution = purchases_df.groupby('Supplier ID')['Count'].count()/11
for staff_id in sorted(purchases_df['Staff ID'].unique()):
    distrib = purchases_df[purchases_df['Staff ID']==staff_id].groupby('Supplier ID')['Count'].count()
    print(distance.jensenshannon(distrib, base_distribution, 2.0))

day_28_df = purchases_df[purchases_df['Day'] == 28]
fig, ax = plt.subplots(nrows=2, ncols=2, sharey=True, figsize=(8,5))
sns.countplot(data=day_28_df, x = 'Supplier ID', color='blue', ax=ax[0][0])
sns.countplot(data=day_28_df[day_28_df['Staff ID']==0], x = 'Supplier ID', color='blue', ax=ax[0][1])
sns.countplot(data=day_28_df[day_28_df['Staff ID']==1], x = 'Supplier ID', color='blue', ax=ax[1][0])
sns.countplot(data=day_28_df[day_28_df['Staff ID']==10], x = 'Supplier ID', color='blue', ax=ax[1][1])
plt.tight_layout()
plt.show()

purchases_df = purchases_df.sort_values(['Date', 'Hour', 'Minute'])
purchases_df['Avg Unit Cost Prev 10'] = purchases_df['Unit Cost'].rolling(window=10).mean()
purchases_df['Daily Cummulative Total Cost'] = purchases_df.groupby('Date')['Total Cost'].cumsum()

purchases_df = purchases_df.sort_values(['Product ID', 'Date', 'Hour', 'Minute'])
purchases_df['Rolling Mean'] = purchases_df.groupby('Product ID')['Unit Cost'].rolling(window=10).mean().values

det = IsolationForest(random_state=0)
df = purchases_df.copy()
subspace_df = df[['Unit Cost', 'Total Cost', 'Count']]
det.fit(subspace_df)
df['IF Scores'] = det.decision_function(subspace_df)
df = df.sort_values(['IF Scores'])
df.head(10)['Staff ID'].value_counts()

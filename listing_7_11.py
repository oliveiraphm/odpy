from listing_7_2 import create_four_clusters_test_data
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler


class EntropyOutlierDetector:
  def __init__(self, num_outliers, num_bins=7):
    self.num_outliers = int(num_outliers)
    self.num_bins = num_bins

  def calculate_entropy(self, values): 
      vc = values.value_counts(normalize=True)
      entropy = 0.0
      for v in vc.values:
          if v > 0:
              entropy += ((v) * np.log2(v))
      entropy = (-1) * entropy
      return entropy

  def fit_predict(self, df):
    df = df.copy()
    df['A binned'] = pd.cut(df[df.columns[0]], bins=self.num_bins) 
    df['B binned'] = pd.cut(df[df.columns[1]], bins=self.num_bins)

    temp_df = df.copy()
    scores = [0]*len(df)
    for iteration_num in range(self.num_outliers): 
        lowest_entropy = np.inf
        lowest_entropy_row = -1
        for i in temp_df.index: 
            a_entropy = self.calculate_entropy(temp_df['A binned'].drop(i))
            b_entropy = self.calculate_entropy(temp_df['B binned'].drop(i))
            total_entropy = a_entropy + b_entropy 
            if total_entropy < lowest_entropy:
                lowest_entropy = total_entropy
                lowest_entropy_row = i

        scores[lowest_entropy_row] = (self.num_outliers - iteration_num)
        temp_df = temp_df.drop(index=lowest_entropy_row)
    return scores

df = create_four_clusters_test_data()
df = pd.DataFrame(RobustScaler().fit_transform(df), columns=df.columns)
df['Entropy Scores'] = EntropyOutlierDetector(num_outliers=10, num_bins=10).fit_predict(df)

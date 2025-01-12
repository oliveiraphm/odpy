from pyod.utils.utility import get_optimal_n_bins
from listing_7_2 import create_four_clusters_test_data

df = create_four_clusters_test_data()
print(get_optimal_n_bins(df))
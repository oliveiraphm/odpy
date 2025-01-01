import numpy as np
import pandas as pd

def create_simple_test_data():
    np.random.seed(0)
    a_data = np.random.normal(size=100)
    b_data = np.random.normal(size=100)
    df = pd.DataFrame({"A": a_data, "B": b_data})
    return df
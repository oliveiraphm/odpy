import numpy as np
import pandas as pd

def create_four_clusters_test_data():
    np.random.seed(0)

    a_data = np.random.normal(loc=25.0, scale=2.0, size=5)
    b_data = np.random.normal(loc=4.0, scale=2.0, size=5)
    df0 = pd.DataFrame({"A": a_data, "B": b_data})

    a_data = np.random.normal(loc=1.0, scale=2.0, size=50)
    b_data = np.random.normal(loc=19.0, scale=2.0, size=50)
    df1 = pd.DataFrame({"A": a_data, "B": b_data})

    a_data = np.random.normal(loc=1.0, scale=1.0, size=200)
    b_data = np.random.normal(loc=1.0, scale=1.0, size=200)
    df2 = pd.DataFrame({"A": a_data, "B": b_data})

    a_data = np.random.normal(loc=20.0, scale=3.0, size=500)
    b_data = np.random.normal(loc=13.0, scale=3.0, size=500) + a_data
    df3 = pd.DataFrame({"A": a_data, "B": b_data})
    
    outliers = [ [ 5.0, 40 ],
                 [ 1.5, 8.0 ],
                 [ 11.0, 0.5 ]
    ]
    df4 = pd.DataFrame(outliers, columns=['A', 'B'])

    df = pd.concat([df0, df1, df2, df3, df4])
    df = df.reset_index(drop=True)
    return df
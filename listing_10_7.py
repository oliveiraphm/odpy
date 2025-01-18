from datasketch.hnsw import HNSW
import numpy as np
data = np.random.random_sample((1000,10))
index = HNSW(distance_func=lambda x, y: np.linalg.norm(x-y))
for i, d in enumerate(data):
    index.insert(i, d)

index.query(data[0], k=10)
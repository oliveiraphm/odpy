from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

row_1 = [1, 0, 0, 0, 0, 0.1]
row_2 = [0, 1, 0, 0, 0, 0.1]

print(manhattan_distances([row_1], [row_2]))    
print(euclidean_distances([row_1], [row_2]))

row_1 = [1, 0, 0.1]
row_2 = [0, 1, 0.1] 

print(manhattan_distances([row_1], [row_2]))
print(euclidean_distances([row_1], [row_2]))

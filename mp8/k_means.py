from copy import deepcopy
import numpy as np
import pandas as pd
import sys


'''
In this problem you write your own K-Means
Clustering code.

Your code should return a 2d array containing
the centers.

'''
# Import the dataset
df = pd.read_csv('./data/data/iris.data')
data = df.iloc[:,:4].as_matrix()

# Make 3  clusters
k = 3

# Initial Centroids
C = [[2.,  0.,  3.,  4.], [1.,  2.,  1.,  3.], [0., 2.,  1.,  0.]]
print("Initial Centers")
print(C)

def k_means(C):
    # Write your code here!
    C = np.array(C)
    while(True):
        old_C = C.copy()
        dist = np.zeros((data.shape[0],k))
        for i in range(k):
            dist[:,i] = np.linalg.norm(data - C[i,:], axis=1)

        min_idx = np.argmin(dist, axis=1)

        for i in range(k):
            C[i,:] = np.average(data[min_idx == i], axis=0)

        if np.array_equal(old_C, C):
            break

    C_final = C.copy()
    return C_final


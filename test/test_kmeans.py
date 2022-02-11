# Write your k-means unit tests here
from cluster import *
import numpy as np

def test_cluster():
    '''
    Test different # of iterations
    and different # of K compared to the ground truth k (k=3)
    '''
    data = np.array([[0, 0],
                 [1, 1],
                 [1, 0],
                 [0, 1],
                 [20, 20],
                 [21, 21],
                 [19, 19],
                 [18, 18],
                 [40, 40],
                 [41,41],
                 [42,42],
                 [42,41],
                 [40,41]])
    k = KMeans(k=3, max_iter=2)
    k2 = KMeans(k=3, max_iter=5)
    k.fit(data)
    k2.fit(data)
    assert k2.get_error() <= k.get_error()
    k3 = KMeans(k=4, max_iter=5)
    k3.fit(data)
    assert k2.get_error() <= k3.get_error()
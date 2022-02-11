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
                 [30, 20],
                 [31, 21],
                 [39, 19],
                 [38, 18],
                 [40, 40],
                 [41,41],
                 [42,42],
                 [42,41],
                 [40,41]])
    k = KMeans(k=2, max_iter=3)
    k2 = KMeans(k=2)
    k.fit(data)
    k2.fit(data)
    score = k.get_error()
    score2 = k2.get_error()
    assert score >= score2
    k3 = KMeans(k=3)
    k4 = KMeans(k=6)
    k3.fit(data)
    k4.fit(data)
    score3 = k3.get_error()
    score4 = k4.get_error()
    assert score3 >= score4

# write your silhouette score unit tests here
from cluster import *
import numpy as np


def test_silhouette_init():
    sil = Silhouette()
    assert sil._metric == "euclidean"


def test_silhouette_score():
    '''
    Tests on ground truth labelling of three different clusters
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
    lab = np.array([0,0,0,0,1,1,1,1,2,2,2,2,2])
    lab2 = np.array([0,0,0,0,0,0,0,1,2,2,2,2,2])
    sil = Silhouette()
    score = sil.score(data,lab)
    score2 = sil.score(data,lab2)
    assert np.nansum(score) > np.nansum(score2)
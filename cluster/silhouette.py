import numpy as np
from scipy.spatial.distance import cdist

class Silhouette:
    def __init__(self, metric: str = "euclidean"):
        """
        inputs:
            metric: str
                the name of the distance metric to use
        """
        self._metric = metric

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features. 

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        mat = np.zeroes(X.shape[0])
        dist = cdist(X,X,self._metric)
        for i in range(X.shape[0]):
            intra_dist =  dist[i, y == y[i]]
            pairwise_intra_dist = np.sum(intra_dist)/(np.sum(y == y[i])-1)
            inter_dist = np.ones(np.max(y))*np.inf
            for j in range(np.max(y)):
                if j != y[j]:
                    inter_dist[j] = np.sum(inter_dist[j,y==j])/np.sum(y==j)
            pairwise_inter_dist = np.min(inter_dist)
            mat[i] = (pairwise_inter_dist-pairwise_intra_dist)/np.max([pairwise_intra_dist, pairwise_inter_dist])
        return mat

import numpy as np
from scipy.spatial.distance import cdist

class KMeans:
    def __init__(
            self,
            k: int,
            metric: str = "euclidean",
            tol: float = 1e-6,
            max_iter: int = 100):
        """
        inputs:
            k: int
                the number of centroids to use in cluster fitting
            metric: str
                the name of the distance metric to use
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        if k == 0:
            raise ValueError("K = 0, K has to K > 0")
        self._k = k
        self._metric = metric
        self._tol = tol
        self._max_iter = max_iter
        self._centroids = None
        self._error = np.inf
    
    def fit(self, mat: np.ndarray):
        """
        fits the kmeans algorithm onto a provided 2D matrix

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        if self._k > mat.shape[0]:
            raise ValueError("More clusters than input entries. Rerun with K less than number of entries.")
        iter = 0
        self._centroids = mat[np.random.choice(mat.shape[0], self._k, replace=False),:]
        while iter < self._max_iter:
            dist = cdist(mat, self._centroids, metric=self._metric)
            centroids = np.argmin(dist, axis=1)
            for i in range(self._k): # Assign centroids with labels
                self._centroids[i,:] = np.mean(mat[centroids==i], axis=0)
            # MSE Mean(Sqaured(Error# min error of dist))
            error = np.average(np.square(np.min(np.square(dist), axis=1)))
            if np.isclose(self._error, error, self._tol):
                self._error = error
                break
            self._error = error
            iter += 1


    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        predicts the cluster labels for a provided 2D matrix

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        return np.argmin(cdist(mat, self._centroids, self._metric), axis = 1)

    def get_error(self) -> float:
        """
        returns the final squared-mean error of the fit model

        outputs:
            float
                the squared-mean error of the fit model
        """
        return self._error

    def get_centroids(self) -> np.ndarray:
        """
        returns the centroid locations of the fit model

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        return self._centroids

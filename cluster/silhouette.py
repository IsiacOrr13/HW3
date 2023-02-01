import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        self.val = 0

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        #s = (distance to nearest cluster) - (average distance to other points in cluster) / (max those)
        scores = np.zeros(shape=(len(X), 1))
        centroids = self._get_centroids(X, y)
        bs = []
        for idx, ele in enumerate(X):
            dists = []
            cluster = y[idx]
            mask = np.array(y) == cluster
            is_k = X[mask]
            for i in is_k:
                d = np.sqrt((i[0] - ele[0])**2 + (i[1] - ele[1])**2)
                dists.append(d)
            a = np.sum(dists) / len(dists)
            for c in centroids:
                bs.append(np.sqrt((c[0] - ele[0]) ** 2 + (c[1] - ele[1]) ** 2))
            b = min(bs)
            scores[idx] = (b - a) / max(a, b)
        return scores


    def _get_centroids(self, X, y) -> np.ndarray:
        k = len(np.unique(y))
        centroids = np.zeros(shape=(k, 2))
        for i in range(k):
            mask = np.array(y) == i
            in_k = X[mask]
            length = in_k.shape[0]
            sum_x = np.sum(in_k[:, 0])
            sum_y = np.sum(in_k[:, 1])
            centroids[i] = [sum_x / length, sum_y / length]
        return centroids






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

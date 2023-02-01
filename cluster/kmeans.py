import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        self._k = k
        self._tol = tol
        self._max_iter = max_iter
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """

    def fit(self, mat: np.ndarray):
        k_idx = np.array(np.random.choice(len(mat), self._k))
        centroids = np.zeros(shape=(self._k, 2))
        for idx, k in enumerate(k_idx):
            centroids[idx] = mat[k]
        del_error = 1
        error = 0
        prev_error = 0
        counter = 0
        while del_error > self._tol and counter < self._max_iter:
            labels = []
            k_dist = cdist(centroids, mat)
            counter += 1
            for i in range(len(mat)):
                dists = k_dist[:,i]
                lab = np.argmin(dists)
                labels.append(lab)
            centroids = self.get_centroids(mat, labels, centroids)
            prev_error = error
            error = self.get_error(mat, labels, centroids)
            del_error = abs(error - prev_error)
        return labels


        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """

    def get_error(self, mat, labels, centroids) -> float:
        dist_mat = cdist(centroids, mat)
        error = 0
        for idx, ele in enumerate(mat):
            curr_label = labels[idx]
            error += dist_mat[curr_label,idx]
        return error


    def get_centroids(self, mat, labels, centroids) -> np.ndarray:
        for i in range(self._k):
            mask = np.array(labels) == i
            in_k = mat[mask]
            length = in_k.shape[0]
            sum_x = np.sum(in_k[:,0])
            sum_y = np.sum(in_k[:,1])
            centroids[i] = [sum_x/length, sum_y/length]
        return centroids

#centroids are not updating past first element


"""
Returns the centroid locations of the fit model.

outputs:
    np.ndarray
        a `k x m` 2D matrix representing the cluster centroids of the fit model
"""


"""
Returns the final squared-mean error of the fit model. You can either do this by storing the
original dataset or recording it following the end of model fitting.

outputs:
    float
        the squared-mean error of the fit model
"""
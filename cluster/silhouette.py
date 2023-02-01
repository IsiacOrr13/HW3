import numpy as np
import math

class Silhouette:
    def __init__(self):
        self._val = 0

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        scores = np.zeros(shape=(len(X), 1))
        centroids = self._get_centroids(X, y)
        for idx, ele, in enumerate(X):
            dist_a = []
            dist_b = []
            dist_c = []
            cluster = y[idx]
            mask = y == cluster
            is_k = X[mask]
            for i in is_k:
                da = math.dist(i, ele)
                dist_a.append(da)
            a = np.sum(dist_a)/len(dist_a)
            for c in centroids:
                dist_c.append(np.sqrt(((c[0] - ele[0]) ** 2) + ((c[1] - ele[1]) ** 2)))
            curr_k = min(dist_c)
            nearest_idx = -1
            curr_dist = 1000
            for idx_k, k in enumerate(dist_c):
                if k < curr_dist and k != curr_k:
                    nearest_idx = idx_k
            nearest_mask = y == nearest_idx
            is_nearest = X[nearest_mask]
            for j in is_nearest:
                db = math.dist(j,ele)
                dist_b.append(db)
            b = np.sum(dist_b)/len(dist_b)
            score = (b - a) / max(b, a)
            scores[idx] = score
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

    '''
    bs = []
            a = 0
            b = 0
            counter_i = 0
            counter_j = 0
            dists = 0
            dists_n = 0
            cluster = y[idx]
            #mask = np.array(y) == cluster
            #is_k = X[mask]
            for idx_i, i in enumerate(X):
                if y[idx_i] == cluster:
                    d = np.sqrt((i[0] - ele[0])**2 + (i[1] - ele[1])**2)
                    dists += d
                    counter_i += 1
            a = dists / counter_i
            for c in centroids:
                bs.append(np.sqrt(((c[0] - ele[0]) ** 2) + ((c[1] - ele[1]) ** 2)))
            b_idx = np.argmin(bs)
            bs = np.delete(bs, b_idx)
            b_idx = np.argmin(bs)
            #near_mask = np.array(y) == b_idx
            #is_nearest = X[near_mask]
            for idx_j, j in enumerate(X):
                if idx_j == b_idx:
                    dn = np.sqrt((j[0] - ele[0]) ** 2 + (j[1] - ele[1]) ** 2)
                    dists_n += dn
                    counter_j += 1
            b = dists_n / counter_j

            scores[idx] = (b - a) / max(a, b)
    '''
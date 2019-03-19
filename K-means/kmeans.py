from cluster import cluster
from copy import deepcopy
import numpy as np

class KMeansCluster(cluster):

    def __init__(self, k = 5 , max_iterations = 100):
        self.k = k
        self.max_iterations = max_iterations

    def fit(self, X):
        # random k centroids
        n = X.shape[0]
        c = X.shape[1]
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        centroids = std * np.random.randn(self.k, c) + mean
        # initial clusters and previous centroids
        centroids_pre = np.zeros(centroids.shape)
        clusters = np.zeros(n, dtype=int)
        iter_Num = 0
        while iter_Num < self.max_iterations:
            # calculate and store the index of closest centroid for each data point
            for i in range(n):
                dist_min = np.linalg.norm(X[i] - centroids[0])
                for t in range(self.k):
                    dist = np.linalg.norm(X[i] - centroids[t])
                    if dist <= dist_min:
                        dist_min = dist
                        clusters[i] = t
            # update the old centroids data structure
            centroids_pre = deepcopy(centroids)
            # calculate the k-mean(mean centroid) and update the centroids
            for k in range(self.k):
                temp = X[clusters == k]
                if len(temp) != 0:
                    centroids[k] = np.mean(temp, axis=0)                
            iter_Num += 1
            # check if converge
            if np.linalg.norm(centroids - centroids_pre) == 0:
                break
        return clusters, centroids
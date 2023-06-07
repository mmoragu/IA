import numpy as np


def findClosestCentroids(X, initial_centroids, K):
    cluster = []
    for point in X:
        euc_dist = []
        for j in range(K):
            euc_dist.append(np.linalg.norm(point - initial_centroids[j]))  # calculo de la distancia euclidea al centroid
        cluster.append(np.argmin(euc_dist))

    return cluster

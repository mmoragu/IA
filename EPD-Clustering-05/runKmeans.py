from computeCentroids import computeCentroids
from findClosestCentroids import findClosestCentroids
import numpy as np

from plotClusters import plotClusters


def runKmeans(X, initial_centroids, max_iters, plot=True):

    K = initial_centroids.shape[0]
    centroids = initial_centroids
    idx = np.zeros((X.shape[0],1))

    for i in range(1,max_iters):
        print("\nIteracion ",i)
        idx = findClosestCentroids(X,centroids,K)
        centroids = computeCentroids(X,idx,K)

    if plot:
        plotClusters(X,idx,centroids,initial_centroids)

    return centroids, idx


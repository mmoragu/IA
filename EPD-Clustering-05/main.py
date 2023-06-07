# EPD5: Machine Learning - Clustering: k-means
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from findClosestCentroids import *
from computeCentroids import *
from runKmeans import *


if __name__ == '__main__':
    print("Loading data and setting initial set of centroids\n")
    X = sio.loadmat("ex7data2.mat")['X']
    # print(X.shape)
    # for i in range(len(X)):
    #     plt.scatter(X[i][0], X[i][1], color="blue")
    # plt.show()


    K = 3  # 3 Centroids
    initial_centroids = np.array([[3.0, 3.0], [6.0, 2.0], [8.0, 5.0]])

    cluster = findClosestCentroids(X, initial_centroids, K)
    cluster = np.asarray(cluster)

    centroids=computeCentroids(X,cluster,K)

    max_iters=10
    centroids, cluster= runKmeans(X, initial_centroids, max_iters, plot=True)
    max_iters = 10

    
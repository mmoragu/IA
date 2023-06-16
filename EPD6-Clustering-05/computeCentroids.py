import numpy as np

def computeCentroids(X, idx, K):

    centroids=[]

    for i in range(K):
        arr_aux = []
        for j in range(len(X)):
            if idx[j] == i:
                arr_aux.append(X[j])
        centroids.append(np.mean(arr_aux,axis=0))
    return np.asarray(centroids)
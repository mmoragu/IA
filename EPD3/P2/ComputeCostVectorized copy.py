import numpy as np

def computeCost_Vectorized(X, y, theta):
    return (np.dot((np.dot(X, theta) - y).T, (np.dot(X, theta) - y))) / (2 * len(X))

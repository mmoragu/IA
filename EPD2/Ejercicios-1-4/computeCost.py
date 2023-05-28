import numpy as np


def computeCost(X, y, theta) :
    m = len(X)
    cost = np.sum(np.square(np.dot(X , theta) - y)) / (2 * m)
    
    return cost

import numpy as np


def normalizedTest(X, test_x):
    X_means = np.array(np.mean(X, axis=0))
    X_std = np.array(np.std(X, axis=0))

    result = []
    for index, x in enumerate(test_x):
        result.append((x - X_means[0][index]) / X_std[0][index])

    return result

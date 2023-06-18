from sigmoid import sigmoid

import numpy as np


def predict(theta_matrix, X):

    num_labels=X.shape[1]

    # Initialize predictions
    p = np.zeros(X.shape[0])
    

    # Calculate the probability for each class
    for i in range(num_labels):
        h = sigmoid(np.dot(X, theta_matrix[i]))
        p += h

    # Find the class with the highest probability
    p = p / num_labels
    pred = np.argmax(p, axis=1)

    return pred
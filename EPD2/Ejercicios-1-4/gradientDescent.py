import computeCost
import numpy as np


# Create a function to gradient descent
def gradientDescent(X, y, theta, iterations=1500, learning_rate=0.01):
    m = len(X)  # Number of training examples

    for _ in range(iterations):
        # Calculate the predicted values
        y_pred = np.dot(X, theta)

        # Calculate the error
        error = y_pred - y
        # Update theta using gradient descent
        theta = theta - (learning_rate / m) * np.dot(X.T, error)

    return theta

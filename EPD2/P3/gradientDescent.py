import numpy as np


def gradientDescent(X, y, theta, iterations=1500, learning_rate=0.01):
    m = len(X)  # Number of training examples

    for iter in range(0,iterations):
        # Calculate the predicted values
        y_pred = np.dot(X, theta)

        # Calculate the error
        error = y_pred - y

        loss=np.dot(X.T, error) 
        gradient=(learning_rate  * loss)/ m
        # Update theta using gradient descent
        theta = theta - gradient

    return theta
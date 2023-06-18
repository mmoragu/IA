import numpy as np
from scipy.optimize import minimize

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function_reg(theta, X, y, lambda_):
    m = len(y)
    h = sigmoid(X @ theta)
    J = (-1 / m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h)) + (lambda_ / (2 * m)) * np.sum(np.square(theta[1:]))
    return J

def gradient_reg(theta, X, y, lambda_):
    m = len(y)
    h = sigmoid(X @ theta)
    grad = (1 / m) * (X.T @ (h - y))
    grad[1:] += (lambda_ / m) * theta[1:]
    return grad

def oneVsAll(X, y, num_labels, lambda_):
    m, n = X.shape
    all_theta = np.zeros((num_labels, n + 1))
    X = np.hstack((np.ones((m, 1)), X))

    for i in range(1, num_labels + 1):
        initial_theta = np.zeros(n + 1)
        options = {'maxiter': 50}
        res = minimize(cost_function_reg, initial_theta, args=(X, (y == i).astype(int), lambda_), jac=gradient_reg, method='CG', options=options)
        all_theta[i - 1] = res.x

    return all_theta
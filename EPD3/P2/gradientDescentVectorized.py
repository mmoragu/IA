import numpy as np
import pandas as pd
from ComputeCostVectorized import *

def gradientDescent_Vectorized(X, y, theta,iterations=1500, learning_rate=0.01):
    m = len(X)  # Number of training examples

    current_iter=[]
    current_cost=[]

    for iter in range(0,iterations):
        # Calculate the predicted values
        y_pred = np.dot(X, theta)

        # Calculate the error
        error = y_pred - y

        loss=np.dot(X.T, error) 
        gradient=(learning_rate  * loss)/ m
        # Update theta using gradient descent
        theta = theta - gradient

        current_iter.append(iter)  # Append the iteration to an array
        current_cost.append(computeCost_Vectorized(X, y, theta)) 

    cost_history=pd.DataFrame({'iteracion': current_iter, 'cost': current_cost})

    return theta, cost_history
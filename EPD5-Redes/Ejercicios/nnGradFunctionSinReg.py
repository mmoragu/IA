import numpy as np
import pandas as pd

from nnCostFunctionSinReg import fordward


def nnGradFunctionSinReg(nn_params_ini, input_layer_size, hidden_layer_size, num_labels, X, y):
    # 1: Enrollar
    theta1 = np.reshape(a=nn_params_ini[:hidden_layer_size * (input_layer_size + 1)],
                        newshape=(hidden_layer_size, input_layer_size + 1), order='F')
    theta2 = np.reshape(a=nn_params_ini[hidden_layer_size * (input_layer_size + 1):],
                        newshape=(num_labels, hidden_layer_size + 1), order='F')

    # 2: salida y --> one hot encoding
    y_d = pd.get_dummies(y.flatten())

    # 3: definimos los delta de salida
    delta1 = np.zeros(theta1.shape)
    delta2 = np.zeros(theta2.shape)
    m = len(y)

    # 4: recorro las filas
    for i in range(X.shape[0]):
        # 4.1: Forward propagation
        a1, a2, a3 = fordward(theta1, theta2, X, i)
        h = a3
        # 4.2: Calculo los errores delta
        d3 = h - y_d.iloc[i]
        d2 = np.multiply(theta2.T @ d3, np.multiply(a2, 1 - a2))
        # 4.3: Calculo de las derivadas
        delta1 = delta1 + np.reshape(d2[1:, ], (hidden_layer_size, 1)) @ np.reshape(a1, (1,input_layer_size+1))  # delta1 = delta1 + d2[np.newaxis:1] @ a1[:newaxis]
        delta2 = delta2 + np.reshape(d3.values, (num_labels, 1)) @ np.reshape(a2, (1, hidden_layer_size+1))  # Hay que ajustar las dimensiones
    delta1 /= m
    delta2 /= m

    # 5: Desenrollar
    derivadas_gradiente = np.hstack((delta1.ravel(order='F'), delta2.ravel(order='F')))
    return derivadas_gradiente



import numpy as np
import pandas as pd

def sigmoid(z):
    return 1/(1+np.exp(-z))


def fordward(Theta1,Theta2,X,i):
    # bias + neuronas de la capa 1
    a1 = np.hstack((np.ones(1), X[i]))
    z2 = Theta1 @ a1
    a2 = sigmoid(z2)
    # bias + neuronas de la capa 2
    a2 = np.hstack((np.ones(1), a2))
    z3 = Theta2 @ a2
    # a3 es la salida de la capa 3 (o h)
    a3 = sigmoid(z3)
    return a1, a2, a3

def nnCostFunctionSinReg(nn_params_ini, input_layer_size, hidden_layer_size, num_labels, X, y):

    m = len(y)
    # 1: Enrollar
    theta1 = np.reshape(a = nn_params_ini[:hidden_layer_size*(input_layer_size+1)],
                        newshape= (hidden_layer_size,input_layer_size+1),order='F')
    theta2 = np.reshape(a = nn_params_ini[hidden_layer_size*(input_layer_size+1):],
                        newshape= (num_labels,hidden_layer_size+1),order='F')
    # 2: salida y --> one hot encoding
    y_d = pd.get_dummies(y.flatten())

    # 3: para cada fila:
    suma = 0
    for i in range(X.shape[0]):
        # 3.1: Forward propagation
        a1,a2,a3 = fordward(theta1,theta2,X,i)
        h = a3 # Tiene la sigmoide ya aplicada
        # 3.2: Calcular el coste de la formula
        temp1 = (y_d.iloc[i])*(np.log(h))
        temp2 = (1-y_d.iloc[i]) * (np.log(1-h))
        temp3 = np.sum(temp1+temp2)
        suma = suma + temp3
    J = (np.sum(suma)) / (-m)

    return J
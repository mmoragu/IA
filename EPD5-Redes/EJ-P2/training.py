import numpy as np
import scipy.io as sio
import scipy.optimize as opt

from nnCostFunctionSinReg import nnCostFunctionSinReg
from nnGradFunctionSinReg import nnGradFunctionSinReg

def training(initial_theta1, initial_theta2, X_train, y_train, input_layer_size, hidden_layer_size, num_labels):
  maxiter = 30 # Si tarda demasiado, se puede bajar el número de iteraciones al hacer la prueba inicial para comprobar que el entrenamiento es el adecuado

  # Paso 1: Desenrollar los parámetros con el mismo order con el que se enrollaron
  nn_initial_params = np.hstack((initial_theta1.ravel(order='F'), initial_theta2.ravel(order='F')))

  # Paso 2: Llamada al optimizador avanzado gradiente conjugado con la función: fmin_cg
  nn_params = opt.fmin_cg(maxiter=maxiter, f=nnCostFunctionSinReg, x0=nn_initial_params, fprime=nnGradFunctionSinReg,
                        args=(input_layer_size, hidden_layer_size, num_labels, X_train, y_train.flatten()))
  
  # Paso 3: Enrollar los pesos/parámetros theta1 y theta2 desde la salida del optimizador avanzado (nn_params)
  theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                      (hidden_layer_size, input_layer_size + 1), order = 'F')
  theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], 
                      (num_labels, hidden_layer_size + 1), order = 'F')
  return theta1, theta2
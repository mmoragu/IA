import numpy as np
import scipy.io as sio
import scipy.optimize as opt

from checkNNGradients import checkNNGradients
from nnCostFunctionConReg import nnCostFunctionConReg
from nnGradFunctionSinReg import nnGradFunctionSinReg
from randInitializeWeights import randInitializeWeights
from predict import predict
from split_data import split_data
if __name__ == '__main__':

    # Setup the parameters you will use for this exercise
    input_layer_size = 400 # 20x20 Input Images of Digits
    hidden_layer_size = 25 # 25 hidden units
    num_labels = 10 # 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)

    # Load Training Data
    print("Loading Data ...\n")
    data = sio.loadmat("./Ejercicios/ex4data1.mat")

    X = data['X']
    y = data['y']
    m = X.shape[0]

    #PROBLEMA 2
    Xtrain, Xtest, ytrain ,ytest=split_data(X,y)



    # Load the weights into variables Theta1 and Theta2
    print("Loading Saved Neural Network Parameters ...\n")
    weights = sio.loadmat("./Ejercicios/ex4weights.mat")
    theta1 = weights['Theta1']
    theta2 = weights['Theta2']
    nn_params_ini = np.hstack((theta1.ravel(order='F'), theta2.ravel(order='F'))) # Unroll parameters


    lambda_param = 1
    J = nnCostFunctionConReg(nn_params_ini, input_layer_size, hidden_layer_size, num_labels, Xtrain,ytrain)


    print("\nChecking Backpropagation... \n")
    lambda_param = 1
    checkNNGradients(lambda_param) # Check gradients by running checkNNGradients

    # ================ EJ3. Initializing Parameters ================
    # In this part of the exercise, you will start by
    # implementing a function to initialize the weights of the neural network
    # (randInitializeWeights.py)
    #
    # Check EB T5 Parte II slides: 21, 22
    print("\nInitializating Neural Network Parameters ...\n")
    initial_theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
    initial_theta2 = randInitializeWeights(hidden_layer_size, num_labels)
    # Unroll parameters (a single column vector)
    nn_initial_params = np.hstack((initial_theta1.ravel(order='F'), initial_theta2.ravel(order='F')))

    
    maxiter = 50
    nn_params = opt.fmin_cg(maxiter=maxiter, f=nnCostFunctionConReg, x0=nn_initial_params, fprime=nnGradFunctionSinReg, args=(input_layer_size, hidden_layer_size, num_labels, Xtrain, ytrain.flatten()))

    theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                           (hidden_layer_size, input_layer_size + 1), 'F')
    theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], (num_labels, hidden_layer_size + 1),
                           'F')


    pred = predict(theta1, theta2, Xtest)
    print("Training Set Accuracy: ", np.mean(pred == y.flatten()) * 100)


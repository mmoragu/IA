import numpy as np
import scipy.io as sio
import scipy.optimize as opt
from predict import predict

from oneVsAll import oneVsAll

if __name__ == '__main__':

    # Setup the parameters you will use for this exercise
    input_layer_size = 400 # 20x20 Input Images of Digits
    hidden_layer_size = 25 # 25 hidden units
    num_labels = 10 # 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)

    # Load Training Data
    print("Loading Data ...\n")
    data = sio.loadmat("./P3/ex4data1.mat")

    X = data['X']
    y = data['y']
    m = X.shape[0]

    # Train one-vs-all classifiers
    theta_matrix = oneVsAll(X, y, num_labels, 0)

    # Predict the labels of the test set
    pred = predict(theta_matrix, X)

    # Calculate the accuracy
    accuracy = np.mean(pred == y.flatten()) * 100
    print("Accuracy: ", accuracy)
import numpy as np

from nnCostFunctionSinReg import fordward


def predict(theta1,theta2,X):
    
    arr_h = [] 

    for i in range(X.shape[0]):

        a1,a2,a3 = fordward(theta1,theta2,X,i)
        h = a3

        arr_h.append(h)



    pred = np.argmax(arr_h, axis=1)+1
    return pred
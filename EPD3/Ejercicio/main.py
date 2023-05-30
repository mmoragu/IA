
import pandas as pd
import numpy as np

from readFile import *
from prepareData import *
from ComputeCostVectorized import computeCost_Vectorized
from gradientDescentVectorized import *
from normalizedTest import *
from normalEquations import *
if __name__ == '__main__':

    ## ======================= EJ1. Cargar y visualizar =======================
    X, y = read_file('./ex1data2.txt')
    X_prima =np.matrix(X)

    X, y , theta = prepareData(X,y)

    cost= computeCost_Vectorized(X,y,theta)

    theta = gradientDescent_Vectorized(X,y,theta)
    
    theta_normalEquations=normalEquations(X_prima,y)

    test_X = np.array([[1650, 3]])
    # new_X = featureNormalize(new_X);
    
    test_X_normalized= normalizedTest(X_prima,test_X)

    testX_normaliz_onesColum = np.append(np.array([[1]]),test_X_normalized)

    prediction = np.dot(testX_normaliz_onesColum,theta)

    prediction_normalEquations=np.dot( np.append(np.array([[1]]),test_X),theta_normalEquations)
    print("Fin")


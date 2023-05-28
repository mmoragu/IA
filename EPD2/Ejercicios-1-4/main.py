# EPD2: Machine Learning - Regresión

import pandas as pd
import numpy as np

from readFile import *
from plotData import *
from gradientDescent import *
from computeCost import *
from ComputeCostVectorized import *
# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    ## ======================= EJ1. Cargar y visualizar =======================
    X, y = read_file('./Material_Alumnos/ex1data1.txt')
    X= X.to_numpy()

    zero_colum=np.zeros((X.shape[0],X.shape[1]))
    X= np.append(zero_colum,X,axis=1)
    y=y.to_numpy()

    #build Theta 
    theta=np.zeros((X.shape[1],1))

    # plotData(X,y)
    ## ======================= EJ2. Función de coste =======================
    
    J_base = computeCost(X, y , theta)
    cost_vectorized=computeCost_Vectorized(X,y,theta)

    ## ======================= EJ3. Gradiente =======================

    theta= gradientDescent(X,y, theta)


    # Predict values for population sizes of 35, 000 and 70, 000
    x_35= np.array([[0,35000/10000]])
    x_70= np.array([0,70000/10000])

    prediction_35= (np.dot(x_35,theta)[0] )*10000
    prediction_70=( np.dot(x_70,theta)[0])*10000


    print("Fin")


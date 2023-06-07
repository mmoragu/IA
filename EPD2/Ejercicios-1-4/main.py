# EPD2: Machine Learning - Regresión

import pandas as pd
import numpy as np

from readFile import *
from plotData import *
from gradientDescent import *
from computeCost import *
from ComputeCostVectorized import *
from prepareData import *
# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    ## ======================= EJ1. Cargar y visualizar =======================
    X, y = read_file('./Ejercicios-1-4/ex1data1.txt')
    

    X, y , theta = prepareData(X,y)

    # plotData(X,y)


    J_base = computeCost(X, y , theta)
    cost_vectorized=computeCost_Vectorized(X,y,theta)

    ## ======================= EJ3. Gradiente =======================

    theta= gradientDescent(X,y, theta)


    # Predict values for population sizes of 35, 000 and 70, 000
    x_35= np.array([[1,35000/10000]])
    x_70= np.array([1,70000/10000])

    prediction_35= (np.dot(x_35,theta) )*10000
    prediction_70=( np.dot(x_70,theta))*10000


    print("Fin")


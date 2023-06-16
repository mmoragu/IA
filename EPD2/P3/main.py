# EPD2: Machine Learning - Regresión

import pandas as pd
import numpy as np

from readFile import *
from ComputeCostVectorized import computeCost_Vectorized
from gradientDescent import gradientDescent
from prepareData import *


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    ## ======================= EJ1. Cargar y visualizar =======================
    X, y = read_file('./P3/ex1data2.txt')
    

    X_1,X_2, y , theta_1, theta_2 = prepareData(X,y)

    # plotData(X,y)


    cost_vectorized_1=computeCost_Vectorized(X_1,y,theta_1)
    cost_vectorized_2=computeCost_Vectorized(X_2,y,theta_2)

    # ## ======================= EJ3. Gradiente =======================

    theta_1= gradientDescent(X,y, theta_1)
    theta_2= gradientDescent(X,y, theta_2)

    cost_vectorized_1=computeCost_Vectorized(X_1,y,theta_1)
    cost_vectorized_2=computeCost_Vectorized(X_2,y,theta_2) 



    print("Fin")


# EPD2: Machine Learning - Regresión

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from readFile import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    ## ======================= EJ1. Cargar y visualizar =======================
    X, y = read_file('./Ejercicios-1-4/ex1data1.txt')
    
    regression = LinearRegression()
    regression.fit(X, y)

    theta = regression.coef_

    predictions = regression.predict(X)



    # Predict values for population sizes of 35, 000 and 70, 000
    x_35= np.array([[35000/10000]])
    x_70= np.array([[70000/10000]])

    prediction_35= regression.predict(x_35) * 10000
    prediction_70= regression.predict(x_70) * 10000

    print("Fin")


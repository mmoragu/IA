
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from gradientDescentVectorized import *
from readFile import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

if __name__ == '__main__':

    ## ======================= EJ1. Cargar y visualizar =======================
    X, y = read_file('./ex1data2.txt')
    X_prima =np.matrix(X)

    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.3)

    Xtrain = np.matrix(Xtrain)
    Xtest = np.matrix(Xtest)
    ytrain = np.matrix(ytrain)
    ytrain = np.matrix(ytest)

    #sklearn 
    model = LinearRegression()
    model.fit(Xtrain, ytrain)

    ypred = model.predict(Xtest)

    mse = mean_squared_error(ytest, ypred)
    r2 = r2_score(ytest, ypred)

    #end sklearn

    #prediccion gradidient
    colum_ones=np.ones((Xtrain.shape[0],1))
    Xtrain=np.append(colum_ones,Xtrain,axis=1)
    

    theta=np.zeros((Xtrain.shape[1],1))

    theta, cost_history=gradientDescent_Vectorized(Xtrain, y, theta)

    theta_sklearn=LinearRegression()

    print("fin")
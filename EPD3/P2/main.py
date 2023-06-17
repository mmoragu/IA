
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from gradientDescentVectorized import *
from readFile import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from standarization import standarization
from ComputeCostVectorized import computeCost_Vectorized

if __name__ == '__main__':

    X, y = read_file('./P1/ex1data2.txt')

    X=standarization(X)
    y=standarization(y)

    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.3)

    Xtrain = np.asarray(Xtrain)
    Xtest = np.asarray(Xtest)
    ytrain = np.asarray(ytrain)
    ytest = np.asarray(ytest)

    #sklearn 
    model = LinearRegression()
    model.fit(Xtrain, ytrain)

    ypred = model.predict(Xtest)

    #el error cuadrático medio (MSE)
    mse = mean_squared_error(ytest, ypred)
    #coeficiente de determinación (R2).
    r2 = r2_score(ytest, ypred)

    #end sklearn

    #prediccion gradidient
    colum_ones=np.ones((Xtrain.shape[0],1))
    Xtrain=np.append(colum_ones,Xtrain,axis=1)
    


    theta=np.zeros((Xtrain.shape[1],1))

    theta, cost_history=gradientDescent_Vectorized(Xtrain, ytrain, theta)


    colum_ones = np.ones((Xtest.shape[0],1))
    Xtest=np.append(colum_ones,Xtest,axis=1)

    ypred_gradient=np.dot(Xtest,theta)

    mse_grad = mean_squared_error(ytest, ypred_gradient)
    #coeficiente de determinación (R2).
    r2_grad = r2_score(ytest, ypred_gradient)
    cost_test = computeCost_Vectorized(Xtest,ytest,theta)

    print("fin")
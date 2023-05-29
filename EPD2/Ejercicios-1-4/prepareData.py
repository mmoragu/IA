import numpy as np

def prepareData(X,y):
    X= np.matrix(X)

    zero_colum=np.zeros((X.shape[0],X.shape[1]))
    X= np.append(zero_colum,X,axis=1)
    y=y.to_numpy()

    #build Theta 
    theta=np.zeros((X.shape[1],1))
    return X, y , theta
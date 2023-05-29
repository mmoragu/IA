import numpy as np

def prepareData(X,y):
    X= np.matrix(X)

    ones_colum=np.zeros((X.shape[0],1))
    X= np.append(ones_colum,X,axis=1)
    y=y.to_numpy()

    #build Theta 
    theta=np.zeros((X.shape[1],1))
    return X, y , theta
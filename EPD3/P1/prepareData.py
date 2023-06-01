import numpy as np
from featureNormalize import *
def prepareData(X,y):

    X= np.matrix(X)
    X=featureNormalize(X)
    
    #add comlums of zeros
    ones_colum=np.ones((X.shape[0],1))
    X= np.append(ones_colum,X,axis=1)
    
    y=y.to_numpy()

    #build Theta 
    theta=np.zeros((X.shape[1],1))
    return X, y , theta
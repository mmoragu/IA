import numpy as np

def normalEquations(X,y):
    ones_colum=np.ones((X.shape[0],1))
    X= np.append(ones_colum,X,axis=1)
    
    p1=np.dot(X.T,X)
    inverse= np.linalg.inv(p1)
    p3=np.dot(inverse,X.T)
    theta=np.dot(p3,y)
    return theta

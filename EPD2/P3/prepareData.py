import numpy as np

def prepareData(X,y):
    X= np.matrix(X)

    X_1 =X[:,:1]
    X_2 =X[:,1:2]

    X_1 = np.asarray(X_1)
    X_2 = np.asarray(X_2)

    ones_colum=np.ones((X.shape[0],1))
    X_1= np.append(ones_colum,X_1,axis=1)
    X_2= np.append(ones_colum,X_2,axis=1)

    y=y.to_numpy()

    #build Theta 
    theta_1=np.zeros((X_1.shape[1],1))
    theta_2=np.zeros((X_2.shape[1],1))
    return X_1,X_2, y , theta_1, theta_2
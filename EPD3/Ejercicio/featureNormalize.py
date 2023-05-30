
def featureNormalize(X):
    # normalize each column of X
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X
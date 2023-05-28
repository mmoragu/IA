

import pandas as pd

def read_file(file_name):
    # Reading file with data 
    file = pd.read_csv(file_name)
    X = file.iloc[:,0:-1]
    y =file.iloc[: ,-1:]

    return X, y
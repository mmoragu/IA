import numpy as np

def split_data(X, y, train_frac=0.7):

    # Get the number of examples
    n_examples = X.shape[0]

    # Get the indices of the training examples
    train_idx = np.random.choice(n_examples, size=int(n_examples * train_frac), replace=False)

    # Get the indices of the test examples
    test_idx = np.setdiff1d(np.arange(n_examples), train_idx)

    # Split the data into training and test sets
    X_train = X[train_idx, :]
    y_train = y[train_idx]
    X_test = X[test_idx, :]
    y_test = y[test_idx]

    return X_train, X_test, y_train, y_test
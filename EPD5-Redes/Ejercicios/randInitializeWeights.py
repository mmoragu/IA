import numpy as np


def randInitializeWeights(L_in, L_out):

    # Set the range epsilon.
    epsilon_init = 0.12

    # Initialize the W matrix.
    W = np.zeros((L_out, 1 + L_in))

    # Produce the random values in the range [-epsilon, epsilon].
    W = np.random.rand(L_out, 1 + L_in) * (2 * epsilon_init) - epsilon_init
    return W
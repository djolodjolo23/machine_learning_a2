import numpy as np


def cost_function(Xe, y, beta, N):
    return 1 / N * ((Xe.dot(beta) - y).T.dot(Xe.dot(beta) - y))  # J(beta) = 1/n(Xe(beta) - y)T(Xe(beta) - y)


def normal_equation(Xe, y):
    return np.linalg.inv(Xe.T.dot(Xe)).dot(Xe.T).dot(y)


def gradient_descent(Xe, y, alpha, num_of_iterations, J_values=None):
    beta_starting = np.zeros(Xe.shape[1])
    for i in range(num_of_iterations):
        beta_starting = beta_starting - alpha * \
                        (Xe.T.dot(
                            Xe.dot(beta_starting) - y))  # beta(j + 1) is the new beta, beta(j) is the current beta
        if J_values is not None:
            cost = cost_function(Xe, y, beta_starting, len(y))
            J_values.append(cost)  # we append the cost function to the list
    return beta_starting


def feature_normalization(Xe):
    mu = np.mean(Xe, axis=0)
    sigma = np.std(Xe, axis=0)
    return (Xe - mu) / sigma


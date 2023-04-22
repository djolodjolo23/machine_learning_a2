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


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_cost_function(Xe, y, beta, N): # g is sigmoid function here
    return -1 / N * (y.T.dot(np.log(sigmoid(Xe.dot(beta)))) + (1 - y).T.dot(np.log(1 - sigmoid(Xe.dot(beta)))))


def gradient_descent_logistic(Xe, y, alpha, num_of_iterations, J_values=None):
    beta_starting = np.zeros(Xe.shape[1])
    cost = logistic_cost_function(Xe, y, beta_starting, len(y))
    if J_values is not None:
        J_values.append(cost)
    for i in range(num_of_iterations):
        beta_starting = beta_starting - (alpha / len(y) * Xe.T).dot(sigmoid(Xe.dot(beta_starting)) - y)
        if J_values is not None:
            cost = logistic_cost_function(Xe, y, beta_starting, len(y))
            J_values.append(cost)
    return beta_starting


def mapFeature(X1, X2, D):
    one = np.ones(X1.shape[0])
    Xe = np.c_[one, X1, X2]
    for i in range(2, D + 1):
        for j in range(0, i + 1):
            Xnew = X1 ** (i - j) * X2 ** j
            Xnew = Xnew.reshape(-1, 1)
            Xe = np.append(Xe, Xnew, 1)
    return Xe


def training_errors(Xe_norm, beta, y_train):
    z = np.dot(Xe_norm, beta).reshape(-1, 1)
    p = sigmoid(z)
    pp = np.round(p)
    yy = y_train.reshape(-1, 1)
    return np.sum(yy != pp)
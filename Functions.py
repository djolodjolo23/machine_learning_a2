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


def gradient_descent_logistic(Xe, y, alpha, num_of_iterations):
    betas, J_values = [], []
    beta_starting = np.zeros(Xe.shape[1])
    cost = logistic_cost_function(Xe, y, beta_starting, len(y))
    J_values.append(cost)
    for i in range(num_of_iterations):
        beta_starting = beta_starting - (alpha / len(y) * Xe.T).dot(sigmoid(Xe.dot(beta_starting)) - y)
        betas.append(beta_starting)
        cost = logistic_cost_function(Xe, y, beta_starting, len(y))
        J_values.append(cost)
    return betas, J_values


def mapFeature(X1, X2, D, ones=None):
    if ones is None or ones:
        one = np.ones(X1.shape[0])
        Xe = np.c_[one, X1, X2]
    elif ones is False:
        Xe = np.c_[X1, X2]
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


def feature_selection(x_train, y_train, num_of_features, LinearRegression):
    x = x_train
    models = []
    base = np.ones((x_train.shape[0], 1))
    #M0
    models.append((base, []))
    mse = 0
    index = 0
    base_temp = np.empty((x_train.shape[0], 1))
    for i in range(num_of_features):
        for j in range(num_of_features - i):
            current_feature = x[:, j].reshape(-1,1)
            new_base = np.c_[base, current_feature]
            model = LinearRegression()
            model.fit(new_base, y_train)
            y_pred = model.predict(new_base)
            mse_current = ((y_pred - y_train) ** 2).mean()
            if mse == 0 or mse_current < mse:
                mse = mse_current
                base_temp = new_base
                index = j
            np.delete(new_base, 1, axis=1)
        column_to_find = base_temp[:, -1]
        col_num = None
        for i in range(x_train.shape[1]):
            if np.array_equal(x_train[:, i], column_to_find):
                col_num = i
                break
        models.append((base_temp, models[-1][1] + [col_num]))
        base = base_temp
        x = np.delete(x, index, axis=1)
    return models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data/GPUbenchmark.csv').to_numpy()

# The main objective is to find the hypothesis f(X) = beta0 + beta1X1 + ... + beta6X6, which
# estimates the linear relation between the graphic card properties and the benchmark result.

# 1. Start by normalizing X using Xn = (X - mu) / sigma and plot the dataset

Xe = np.array([np.ones(data.shape[0]), data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5]]).T
y = data[:, 6]

mu = np.mean(Xe, axis=0)
sigma = np.std(Xe, axis=0)

Xn = (Xe - mu) / sigma

# 2. Multivariate datasets are hard to visualize. However, to get a basic understanding it might be a good idea to produce
# a plot Xi vs y for each one of the features. Use subplot(2, 3, i) to fit all six plots into a single figure. Make sure that each
# normalized Xi is centralized around 0.

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
axes = [ax1, ax2, ax3, ax4, ax5, ax6]
for i in range(len(axes)):
    axes[i].scatter(Xn[:, i + 1], y)
    axes[i].set_title('X' + str(i) + ' vs y')
    axes[i].set_xlabel('X' + str(i))
    axes[i].set_ylabel('y')
plt.show()

# 3. Compute beta using the normal equation b = (xe.T * xe)^-1 * xe.T * y where Xe is the extended normalized matrix [1, X1, X2, ... X6].
# What is the predicted benchmark result for a graphic card with the following (non-normalized) feature values?
# 2432, 1607, 1683, 8, 8, 256. The actual benchmark result is 114

Xn[:, 0] = 1

beta_normalized = np.linalg.inv(Xn.T.dot(Xn)).dot(Xn.T).dot(y)
beta = np.linalg.inv(Xe.T.dot(Xe)).dot(Xe.T).dot(y)

non_normalized = np.array([1, 2432, 1607, 1683, 8, 8, 256])
prediction = non_normalized.dot(beta) # prediction is around 110

# 4. What is the cost J(b) when using the beta computed by the normal equation above?

def cost_function(Xe, y, beta, N):
    return 1 / N * ((Xe.dot(beta) - y).T.dot(Xe.dot(beta) - y))

cost = cost_function(Xn, y, beta, len(y)) # cost function gives 12261.47

# 5. a) Find (and print) hyperparameters (alpha, N) such that you get within 1% of the final cost for the normal equation.

beta_gradient = np.zeros(Xn.shape[1])
alpha = 0.001
NumOfIterations = 100

def gradientDescent(Xe, y, beta_starting, alpha, num_of_iterations, J_values):
    for i in range(num_of_iterations):
        beta_starting = beta_starting - alpha * (Xe.T.dot(Xe.dot(beta_starting) - y)) # beta(j + 1) is the new beta, beta(j) is the current beta
        cost = cost_function(Xe, y, beta_starting, len(y)) # we compute the cost function
        J_values.append(cost) # we append the cost function to the list
    return beta_starting


J_values = []
beta_gradient = gradientDescent(Xn, y, beta_gradient, alpha, NumOfIterations, J_values)

# (b) What is the predicted benchmark result for the example graphic card presented

normalize = non_normalized / np.linalg.norm(non_normalized)
prediction_gradient = normalize.dot(beta_gradient) # prediction with normalized data is around 8.71





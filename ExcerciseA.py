import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('data/stat_females.csv', delimiter='\t').to_numpy()

# plot the data

x1 = data[:, 1]
x2 = data[:, 2]
y = data[:, 0]  # girls height

'''''
fig, ax = plt.subplots()
ax.scatter(x1, x2, c=y)
ax.set_xlabel('Dad height')
ax.set_ylabel('Mom height')
ax.set_title("initial plot")
plt.show()
'''
# Compute the extended matrix xe = [1, x1, x2]

Xe = np.array([np.ones(x1.shape[0]), x1, x2]).T

# Implement the normal equation b = (xe.T * xe)^-1 * xe.T * y

beta = np.linalg.inv(Xe.T.dot(Xe)).dot(Xe.T).dot(y)
test_parents = np.array([1, 65, 70])
find_girls_height = test_parents.dot(beta)

# Implement feature normalization Xn = (X - mu) / sigma and plot the dataset
# once again to verify that the mom and dad heights are centered around 0
# with a standard deviation of 1

mu = np.mean(Xe, axis=0)
sigma = np.std(Xe, axis=0)

Xn = (Xe - mu) / sigma

fig, ax = plt.subplots()
ax.scatter(Xn[:, 1], Xn[:, 2], c=y)
ax.set_xlabel('Dad height')
ax.set_ylabel('Mom height')
ax.set_title("normalized plot")
# plt.show()

# 5. Compute the extended matrix Xe and apply the Normal equation on the normalized
# version of (65, 70). The prediction should still be 65.42 inches

Xe2 = np.array([np.ones(x1.shape[0]), Xn[:, 1], Xn[:, 2]]).T
beta2 = np.linalg.inv(Xe2.T.dot(Xe2)).dot(Xe2.T).dot(y)
test_parents2 = np.array([1, (65 - mu[1]) / sigma[1], (70 - mu[2]) / sigma[2]])
find_girls_height2 = test_parents2.dot(beta2)


#  Implement the cost function J(beta) = 1/n(Xe(beta) - y)T(Xe(beta) - y) as a function of parameters
# Xe, y, beta. The cost for beta from the Normal equation should be 4.068

def cost_function(Xe, y, beta):
    return 1 / Xe.shape[0] * (Xe.dot(beta) - y).T.dot(Xe.dot(beta) - y)  # J(beta) = 1/n(Xe(beta) - y)T(Xe(beta) - y)


cost = cost_function(Xe, y, beta) # or beta2

# 7. Gradient descent algorithm beta (j+1) = beta (j) - alpha * Xe.T * (Xe * beta (j) - y)
# a) Implement vectorized gradient descent algorithm





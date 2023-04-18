import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('data/stat_females.csv', delimiter='\t').to_numpy()

# plot the data

x1 = data[:, 1] # feature 1
x2 = data[:, 2] # feature 2
y = data[:, 0]  # girls height or target variable or output

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

beta = np.linalg.inv(Xe.T.dot(Xe)).dot(Xe.T).dot(y) # these are  weights
test_parents = np.array([1, 65, 70])
find_girls_height = test_parents.dot(beta) #prediction for any point

# Implement feature normalization Xn = (X - mu) / sigma and plot the dataset
# once again to verify that the mom and dad heights are centered around 0
# with a standard deviation of 1

mu = np.mean(Xe, axis=0)
sigma = np.std(Xe, axis=0)

Xn = (Xe - mu) / sigma

'''''
fig, ax = plt.subplots()
ax.scatter(Xn[:, 1], Xn[:, 2], c=y)
ax.set_xlabel('Dad height')
ax.set_ylabel('Mom height')
ax.set_title("normalized plot")
# plt.show()
'''
# 5. Compute the extended matrix Xe and apply the Normal equation on the normalized
# version of (65, 70). The prediction should still be 65.42 inches

Xe2 = np.array([np.ones(x1.shape[0]), Xn[:, 1], Xn[:, 2]]).T
beta2 = np.linalg.inv(Xe2.T.dot(Xe2)).dot(Xe2.T).dot(y)
test_parents2 = np.array([1, (65 - mu[1]) / sigma[1], (70 - mu[2]) / sigma[2]])
find_girls_height2 = test_parents2.dot(beta2) # prediction for any point


# 6 Implement the cost function J(beta) = 1/n(Xe(beta) - y)T(Xe(beta) - y) as a function of parameters
# Xe, y, beta. The cost for beta from the Normal equation should be 4.068

def cost_function(Xe, y, beta):
    return 1 / Xe.shape[0] * (Xe.dot(beta) - y).T.dot(Xe.dot(beta) - y)  # J(beta) = 1/n(Xe(beta) - y)T(Xe(beta) - y)


cost = cost_function(Xe, y, beta) # or beta2

# 7. A)Implement the gradient descent algorithm and apply it to the normalized dataset

XeNormalized = np.c_[np.ones(Xe.shape[0]), Xn[:, 1], Xn[:, 2]]
y = y
alpha = 0.001
num_of_iterations = 250

beta_starting = np.zeros(Xe.shape[1])

def gradientDescent(Xe, y, beta_starting, alpha, num_of_iterations, J_values):
    for i in range(num_of_iterations):
        #gradient = Xe.T.dot(Xe.dot(beta_starting) - y) # first we compute the gradient, second part of the formula given in the assignment sheet
        beta_starting = beta_starting - alpha * (Xe.T.dot(Xe.dot(beta_starting) - y)) # first part of the formula given in the assignment sheet, # beta(j + 1) is the new beta, beta(j) is the current beta
        cost = cost_function(Xe, y, beta_starting) # we compute the cost function
        J_values.append(cost) # we append the cost function to the list
    return beta_starting

#beta3 = gradientDescent(XeNormalized, y, beta_starting, alpha, num_of_iterations)



# 7. B) Plot the cost function as a function of the number of iterations

J_values = []

beta4 = gradientDescent(XeNormalized, y, beta_starting, alpha, num_of_iterations, J_values)

plt.plot(range(num_of_iterations), J_values)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()




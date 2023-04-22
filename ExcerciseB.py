import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


import Functions as f
from sklearn.model_selection import train_test_split



data = pd.read_csv('data/admission.csv').to_numpy()

random_state = np.random.randint(1, 301)
# splitting the data into training and testing (80-20)
X_train, X_test, y_train, y_test = train_test_split(data[:, :2], data[:, 2], test_size=0.2, random_state=random_state)

# Normalazing the features
Xn = f.feature_normalization(X_train)

# Plotting the data using the different markes for the two labels
'''''
colors = ListedColormap(['red', 'blue'])
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
ax.scatter(Xn[:, 0], Xn[:, 1], c=y_train, cmap=colors)
ax.set_title('X1 vs X2')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
legend_labels = ['Not admitted', 'Admitted']
legend_handles = [mpatches.Patch(color='red', label='Not admitted'), mpatches.Patch(color='blue', label='Admitted')]
ax.legend(handles=legend_handles, labels=legend_labels)

plt.show()
'''
# implement the sigmoid function that can take any numpy array as input and output the matrix of the same size
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

test_array = np.array([[0,1], [2, 3]])

z = sigmoid(-1)

print(sigmoid(test_array))

# extend X to make it suitable for a linear assumption

Xe = np.array([np.ones(Xn.shape[0]), Xn[:, 0], Xn[:, 1]]).T

# implement a vectorized version of the logistic cost function

def logistic_cost_function(Xe, y, beta, N): # g is sigmoid function here
    return -1 / N * (y.T.dot(np.log(sigmoid(Xe.dot(beta)))) + (1 - y).T.dot(np.log(1 - sigmoid(Xe.dot(beta)))))


zz = logistic_cost_function(Xe, y_train, np.array([0, 0, 0]), len(y_train))

# implement a vectorized version of gradient descent:
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

J_values = []
beta_gradient = gradient_descent_logistic(Xe, y_train, 0.5, 600, J_values)
def mapFeature(x1, x2, degree):
    out = np.ones(x1.shape[0])
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out = np.vstack((out, (x1 ** (i - j)) * (x2 ** j)))
    return out.T

# plot a decision boundary
h = 0.01 # step size in the mesh
x_min, x_max = Xn[:, 0].min() -0.1, Xn[:, 0].max()+0.1
y_min, y_max = Xn[:, 1].min() -0.1, Xn[:, 1].max()+0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
x1, x2 = xx.ravel(), yy.ravel()
Xxe = mapFeature(x1, x2, 2)
p = sigmoid(np.dot(Xxe, beta_gradient))
classes = p > 0.5
clz_mesh = classes.reshape(xx.shape)

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

plt.figure(2)
plt.pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
plt.scatter(Xn[:, 0], Xn[:, 1], c=y_train, cmap=cmap_bold)
plt.show()











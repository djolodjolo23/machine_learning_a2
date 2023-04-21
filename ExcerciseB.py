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

# implement the sigmoid function that can take any numpy array as input and output the matrix of the same size
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

test_array = np.array([[0,1], [2, 3]])
print(sigmoid(test_array))

# extend X to make it suitable for a linear assumption

Xe = np.array([np.ones(Xn.shape[0]), Xn[:, 0], Xn[:, 1]]).T

# implement a vectorized version of the logistic cost function

def logistic_cost_function(Xe, y, beta, N): # g is sigmoid function here
    return -1 / N * (y.T.dot(np.log(sigmoid(Xe.dot(beta)))) + (1 - y).T.dot(np.log(1 - sigmoid(Xe.dot(beta)))))


zz = logistic_cost_function(Xe, y_train, np.array([0, 0, 0]), len(y_train))










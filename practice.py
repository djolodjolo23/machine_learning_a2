import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_csv('secret_polynomial.csv')
x = data['x'].values
y = data['y'].values

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create a list of polynomial degrees to try
degrees = range(1, 7)

# Initialize lists to store evaluation metric values for each degree
mse_train = []
mse_test = []

# Loop through each degree
for degree in degrees:
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    x_train_poly = poly_features.fit_transform(x_train.reshape(-1, 1))
    x_test_poly = poly_features.transform(x_test.reshape(-1, 1))

    # Fit a polynomial regression model
    model = LinearRegression()
    model.fit(x_train_poly, y_train)

    # Predict on training and test data
    y_train_pred = model.predict(x_train_poly)
    y_test_pred = model.predict(x_test_poly)

    # Compute evaluation metric (MSE) for training and test data
    mse_train.append(mean_squared_error(y_train, y_train_pred))
    mse_test.append(mean_squared_error(y_test, y_test_pred))

# Find the best degree with the lowest MSE on test data
best_degree = degrees[np.argmin(mse_test)]

# Plot the fitted polynomial curves along with the training data
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.flatten()
for i, degree in enumerate(degrees):
    poly_features = PolynomialFeatures(degree=degree)
    x_train_poly = poly_features.fit_transform(x_train.reshape(-1, 1))
    x_test_poly = poly_features.transform(x_test.reshape(-1, 1))

    model = LinearRegression()
    model.fit(x_train_poly



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Functions as f
from sklearn.model_selection import train_test_split



data = pd.read_csv('data/admission.csv').to_numpy()

random_state = np.random.randint(1, 301)
# splitting the data into training and testing (80-20)
X_train, X_test, y_train, y_test = train_test_split(data[:, :2], data[:, 2], test_size=0.2, random_state=random_state)

# Normalazing the features
Xn = f.feature_normalization(X_train)

# Plotting the data using the different markes for the two labels

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
ax.scatter(Xn[:, 0], Xn[:, 1], c=y_train, cmap='bwr')
ax.set_title('X1 vs X2')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
plt.show()







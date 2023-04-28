import pandas as pd
import numpy as np
#%% md
#%%
data = pd.read_csv('data/cars-mpg.csv').to_numpy()
# split the x_train, y_train, x_test, y_test with sklearn train test split, first row is the label
from sklearn.model_selection import train_test_split
random_state = np.random.seed(1)
x_train, x_test, y_train, y_test = train_test_split(data[:,1:], data[:,0], test_size=0.2, random_state=random_state)


from sklearn.linear_model import LinearRegression
num_of_features = x_train.shape[1]


# models should be created with the number of features
# m1 should be a model with only one feature
# m2 should be a model with two features
# m3 should be a model with three features...


def feature_selection():
    models = []
    base = np.ones((x_train.shape[0], 1))
    models.append(base)
    Model0 = LinearRegression()
    Model0.fit(base, y_train)
    y_pred = Model0.predict(base)
    mse = ((y_pred - y_train) ** 2).mean()
    base_temp = np.empty((x_train.shape[0], 1))
    for i in range(num_of_features):
        for j in range(num_of_features):
            current_feature = x_train[:, j].reshape(-1,1)
            new_base = np.c_[base, current_feature]
            Model = LinearRegression()
            Model.fit(new_base, y_train)
            y_pred = Model.predict(new_base)
            mse_current = ((y_pred - y_train) ** 2).mean()
            if mse_current < mse:
                mse = mse_current
                base_temp = new_base
            np.delete(new_base, 1, axis=1)
        models.append(base_temp)
        base = base_temp
    return models


models = feature_selection()
print(models[0].shape)


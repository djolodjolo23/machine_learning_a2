
import pandas as pd
import numpy as np
import Functions as f
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#%%
gpus = pd.read_csv('data/GPUbenchmark.csv').to_numpy()
random_state = np.random.seed(1)
x_train, x_val, y_train, y_val = train_test_split(gpus[:,:6], gpus[:,-1], test_size=0.2, random_state=random_state)
num_of_features = x_train.shape[1]

models = f.feature_selection(x_train, y_train, num_of_features, LinearRegression)

zz = models[2]

print(zz)
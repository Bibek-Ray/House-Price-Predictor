"""
## Importing the libraries
"""
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso, Ridge

lasso_model = Lasso(alpha=73.0)
ridge_model = Ridge(alpha=73.0)

"""## Importing the dataset"""
import sys
np.set_printoptions(threshold=sys.maxsize)
dataset = pd.read_csv('Housing.csv')
x = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values
x1 = dataset.iloc[: , 5:10].values # 5th to 9th column selected (they are not integers)
x2 = dataset.iloc[: , [11]].values # 11th column selected (they are not integers)
x8 = dataset.iloc[: , [12]].values # 12th column selected (they are not integers)
x3 = dataset.iloc[: , 10:11].values # 10th column is integer
x4 = dataset.iloc[: , 1:5].values

"""# Encoding non numeric data to numeric type"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct1 = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x2 = np.array(ct1.fit_transform(x2))
#ct3 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')

# Manually encode the 'x8' column
encoded_values = []
for value in x8:
    if value == 'furnished':
        encoded_values.append([1, 0, 0])
    elif value == 'semi-furnished':
        encoded_values.append([0, 1, 0])
    elif value == 'unfurnished':
        encoded_values.append([0, 0, 1])

x8_encoded = np.array(encoded_values)
x8 = x8_encoded

x2 = np.concatenate((x2, x8), axis=1)
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0, 1, 2, 3, 4])], remainder='passthrough')
x1 = np.array(ct.fit_transform(x1))
x5 = np.concatenate((x1, x3), axis=1)
x6 = np.concatenate((x5, x2), axis=1)
x7 = np.concatenate((x4, x6), axis=1)
x= x7

"""## Splitting the dataset into the Training set and Test set"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 0)

lasso_model.fit(X_train, y_train)  # For L1 regularization (Lasso)

y_pred_lasso = lasso_model.predict(X_test)  # Predict using L1 regularization (Lasso)

# Evaluate the performance of the models
from sklearn.metrics import mean_squared_error

rmse_lasso = mean_squared_error(y_test, y_pred_lasso, squared=False)  # RMSE for L1 regularization (Lasso)

print(rmse_lasso)

from sklearn.metrics import r2_score
v = r2_score(y_test, y_pred_lasso)
adj_r2 = 1 - (1 - v)*((len(y_pred_lasso) - 1)/(len(y_pred_lasso) - 13))

print(adj_r2)

pickle.dump(lasso_model , open('model.pkl', 'wb'))
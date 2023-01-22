# -*- coding: utf-8 -*-
"""
## Importing the libraries
"""

import numpy as np
import pandas as pd

"""## Importing the dataset"""

dataset = pd.read_csv('Housing.csv')
x = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values
x1 = dataset.iloc[: , 5:10].values
x2 = dataset.iloc[: , 11:].values
x3 = dataset.iloc[: , 10:11].values
x4 = dataset.iloc[: , 0:5].values

"""# Encoding non numeric data to numeric type"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct1 = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0, 1])], remainder='passthrough')
x2 = np.array(ct1.fit_transform(x2))
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0, 1, 2, 3, 4])], remainder='passthrough')
x1 = np.array(ct.fit_transform(x1))
x5 = np.concatenate((x1, x3), axis=1)
x6 = np.concatenate((x5, x2), axis=1)
x7 = np.concatenate((x4, x6), axis=1)
x= x7

"""## Splitting the dataset into the Training set and Test set"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

"""## Training the Random Forest Regression model on the whole dataset"""

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

"""## Predicting the Test set results"""

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

"""## Evaluating the Model Performance"""

from sklearn.metrics import r2_score
v = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - v)*((len(y_pred) - 1)/(len(y_pred) - 13))

print(adj_r2)

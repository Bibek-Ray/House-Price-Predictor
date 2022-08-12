import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

count = 0
dataset = pd.read_csv('Housing.csv')
x = dataset.iloc[: , 1: ].values
y = dataset.iloc[:, :1].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [4, 5, 6, 7, 8, 10, 11])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression as LR
regressor = LR()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
np.set_printoptions(precision = 2) #This statement restricts more than 2 values after decimal
#print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
for i in range(len(y_test)):
    if((y_pred[i] <= (1.2 * y_test[i])) and (y_pred[i]) >= (0.8 * y_test[i] )):
        count += 1
    Percent = (count/len(y_test)) * 100
print(Percent)
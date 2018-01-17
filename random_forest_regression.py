# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:30:55 2018

@author: Jaskaran
"""

# Random Forest Regression

# Importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Position_Salaries.csv')

# pd.set_option('display.float_format', lambda x: '%.0f' % x)

# matrix of features (independent variables)
X = dataset.iloc[:, 1:2].values
# dependant variable vector
y = dataset.iloc[:, -1].values

"""# Splitting the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

""" # Feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train) """


# Fitting the Random Forest Regression Model to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X, y)

# predicting new result
y_pred = regressor.predict(6.5)

# Visualize the results
# for higher resolution and smoother curve
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'green')
plt.title('Random Forest Regression plot')
plt.xlabel('Position level')
plt.ylabel('Salary')
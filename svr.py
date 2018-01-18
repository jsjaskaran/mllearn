# -*- coding: utf-8 -*-
"""
@author: Jaskaran
"""
# SVR

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

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


# Fitting the SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# predicting new result
# we are using inverse transform to not get scaled value
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
print y_pred
# Visualize the results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'green')
plt.title('Polynomial Regression plot')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# for higher resolution and smoother curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'green')
plt.title('SVR plot')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
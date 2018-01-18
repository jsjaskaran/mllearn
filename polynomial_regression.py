# Polynomial Regression

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

# Fitting Linear regression
from sklearn.linear_model import LinearRegression

linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

# Fitting Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures

poly_regressor = PolynomialFeatures(degree = 4)
X_poly = poly_regressor.fit_transform(X)

lin_poly_reg = LinearRegression()
lin_poly_reg.fit(X_poly, y)

# Visualize the results
# 1. 
plt.scatter(X, y, color = 'red')
plt.plot(X, linear_regressor.predict(X), color = 'green')
plt.title('Linear Regression plot')
plt.xlabel('Position level')
plt.ylabel('Salary')

# 2.
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_poly_reg.predict(poly_regressor.fit_transform(X_grid)), color = 'green')
plt.title('Polynomial Regression plot')
plt.xlabel('Position level')
plt.ylabel('Salary')

# Predicting new results

# with Linear Regression
linear_regressor.predict(6.5)

# with Polynomial Regression
lin_poly_reg.predict(poly_regressor.fit_transform(6.5))
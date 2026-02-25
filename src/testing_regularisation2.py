from .linear_regression import LinearRegression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# Scaling is important before regularization

housing = fetch_california_housing()
scaler = StandardScaler()

X = pd.DataFrame(housing.data, columns=housing.feature_names)[["AveRooms", "MedInc", "HouseAge", "AveBedrms", "AveOccup"]]
y = housing.target
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)

ridge_reg = LinearRegression(ridge_alpha=0.1)
ridge_reg.fit(X_train, y_train)

print("Ridge Regression:")
print(ridge_reg.coef_)
print(ridge_reg.intercept_)

lasso_reg = LinearRegression(lasso_alpha=0.1, l1_ratio=1)
lasso_reg.fit(X_train, y_train)

print("Lasso Regression:")
print(lasso_reg.coef_)
print(lasso_reg.intercept_)

elastic_net = LinearRegression(ridge_alpha=0.1, lasso_alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X_train, y_train)

print("ElasticNet Regression:")
print(elastic_net.coef_)
print(elastic_net.intercept_)

# Converged at  53618 th step
# Ridge Regression:
# [-0.49522355  1.0059868   0.200353    0.45162886 -0.04509436]
# 2.066487821162555
# Converged at  5630 th step
# Lasso Regression:
# [ 6.17906680e-05  7.06624121e-01  1.06936025e-01  7.16709939e-06 -1.14529272e-04]
# 2.060571255333859
# Converged at  4377 th step
# ElasticNet Regression:
# [ 5.70325816e-06  7.55822341e-01  1.66388294e-01 -1.64102359e-05 -2.60158577e-05]
# 2.044141572556723

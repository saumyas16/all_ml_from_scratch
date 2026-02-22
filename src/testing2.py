from .linear_regression import LinearRegression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)[["AveRooms", "MedInc", "HouseAge", "AveBedrms", "AveOccup"]]
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

print(lin_reg.coef_)
print(lin_reg.intercept_)

y_pred = lin_reg.predict(X_test)
print(y)
print(y_pred)

# AveRooms    -0.179497
# MedInc       0.515162
# HouseAge     0.015323
# AveBedrms    0.825322
# AveOccup    -0.004365
# -0.2803297172341129

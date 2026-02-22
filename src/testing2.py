from .regression import linear_regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)[["AveRooms", "MedInc", "HouseAge", "AveBedrms", "AveOccup"]]
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)

coef, intercept, mse = linear_regression(X_train, y_train)

print(coef)
print(intercept)

# AveRooms    -0.179497
# MedInc       0.515162
# HouseAge     0.015323
# AveBedrms    0.825322
# AveOccup    -0.004365
# -0.2803297172341129

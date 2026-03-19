from src.lib.gradient_descent_backtacking import LinearRegression as BackTrackLR

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)[["AveRooms", "MedInc", "HouseAge", "AveBedrms", "AveOccup"]]
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)

lin_reg = BackTrackLR(rate=0.05)
lin_reg.fit(X_train, y_train)

print(lin_reg.coef_)
print(lin_reg.intercept_)

y_pred = lin_reg.predict(X_test)
print(y)
print(y_pred)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

print(lin_reg.coef_)
print(lin_reg.intercept_)

y_pred = lin_reg.predict(X_test)
print(y)
print(y_pred)

# AveRooms    -0.175458
# MedInc       0.510421
# HouseAge     0.014892
# AveBedrms    0.794860
# AveOccup    -0.004392
# dtype: float64
# -0.23595564691894638

# [-0.2092519   0.53435117  0.01585954  0.99778784 -0.0043579 ]
# -0.40084217884263573

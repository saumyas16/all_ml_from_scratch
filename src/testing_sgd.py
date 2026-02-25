import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from .sgd_linear_regression import SGDLinearRegression

housing = fetch_california_housing()
scaler = StandardScaler()

X = pd.DataFrame(housing.data, columns=housing.feature_names)[["AveRooms", "MedInc", "HouseAge", "AveBedrms", "AveOccup"]]
y = housing.target
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5, penalty=None, eta0=0.01,
                       n_iter_no_change=100, random_state=42)
sgd_reg.fit(X_train, y_train)

print(sgd_reg.coef_)
print(sgd_reg.intercept_)

my_sgd_reg = SGDLinearRegression(max_iter=1000)
my_sgd_reg.fit(X_train, y_train)

print(my_sgd_reg.coef_)
print(my_sgd_reg.intercept_)

# sklearn SGD
# [-0.49586657  1.01475242  0.20291029  0.43221173 -0.07222662]
# [2.06721115]

# My SGD
# [-0.50059082  1.0054433   0.20005729  0.45445414 -0.04658337]
# 2.0646071867027715

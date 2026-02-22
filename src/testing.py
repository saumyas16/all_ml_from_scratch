import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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

# [-0.2092519   0.53435117  0.01585954  0.99778784 -0.0043579 ]
# -0.40084217884263573

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from src.lib.AdaBoostRegressor import AdaBoostRegressor as myABR
from src.lib.CART_regressor import DecisionTreeRegressor as myDTR

X, y = make_regression(n_samples=1000, n_features=4, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

ada_reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=3), n_estimators=40, learning_rate=1.0)
ada_reg.fit(X_train, y_train)

my_ada_reg = myABR(regressor=myDTR(max_depth=3), learning_rate=1.0, n_estimators=40)
my_ada_reg.fit(X_train, y_train)

print(y_test[:4])
print(ada_reg.predict(X_test[:4]))
print(my_ada_reg.predict(X_test[:4]))

sk_y_pred = ada_reg.predict(X_test)
my_y_pred = my_ada_reg.predict(X_test)

print(mean_squared_error(y_test, sk_y_pred))
print(mean_squared_error(y_test, my_y_pred))

print(r2_score(y_test, sk_y_pred))
print(r2_score(y_test, my_y_pred))

# [-144.99225315   27.43026212  -38.52182337   -6.70497312]
# [-104.7727254    17.73234471  -25.50263025  -18.95146964]
# [-97.98803527  19.51585681  -0.13069889 -50.56238975]
# 1285.4795944021625
# 1444.5453305002022
# 0.8973099555901844
# 0.8846030502646399

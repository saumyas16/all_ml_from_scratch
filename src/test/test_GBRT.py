import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from src.lib.GradientBoostRegressor import GradientBoostRegressor as myGBRT

m = 100
rng = np.random.default_rng(seed=30)
X = rng.random((m, 1)) - 0.5
noise = 0.05 * rng.standard_normal(m)
y = 3 * X[:, 0] ** 2 + noise

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)
gbrt.fit(X, y)

my_gbrt = myGBRT(n_estimators=3, max_depth=2, learning_rate=1.0)
my_gbrt.fit(X, y)

print(gbrt.predict(X[:4]))
print(my_gbrt.predict(X[:4]))

gbrt2 = GradientBoostingRegressor(max_depth=2, n_estimators=500, learning_rate=0.05, n_iter_no_change=10, random_state=42)
gbrt2.fit(X, y)
my_gbrt2 = myGBRT(n_estimators=500, max_depth=2, learning_rate=0.05, n_iter_no_change=10, random_state=42)
my_gbrt2.fit(X, y)

gbrt3 = GradientBoostingRegressor(max_depth=2, n_estimators=500, learning_rate=0.05, n_iter_no_change=10, random_state=42, subsample=0.25)
gbrt3.fit(X, y)
my_gbrt3 = myGBRT(n_estimators=500, max_depth=2, learning_rate=0.05, n_iter_no_change=10, random_state=42, subsample=0.25)
my_gbrt3.fit(X, y)

print(gbrt2.predict(X[:4]))
print(gbrt2.n_estimators_)
print(my_gbrt2.predict(X[:4]))
print(my_gbrt2.n_estimators_)

print(gbrt3.predict(X[:4]))
print(gbrt3.n_estimators_)
print(my_gbrt3.predict(X[:4]))
print(my_gbrt3.n_estimators_)

# [0.32784435 0.02764544 0.43929862 0.02764544]
# [0.32784435 0.02764544 0.43929862 0.02764544]
# [0.23753273 0.03021634 0.43896946 0.03021634]
# 83
# [0.1930453  0.03249206 0.45546829 0.03249206]
# 89
# [0.25378994 0.0430671  0.40215049 0.04788272]
# 56
# [0.22143108 0.03978558 0.4332719  0.0341091 ]
# 70

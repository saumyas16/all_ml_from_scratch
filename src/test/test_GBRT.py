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

# [0.32784435 0.02764544 0.43929862 0.02764544]
# [0.32784435 0.02764544 0.43929862 0.02764544]

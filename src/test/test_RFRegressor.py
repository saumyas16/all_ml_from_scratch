from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from src.lib.RandomForestRegressor import RandomForestRegressor as myRF

X, y = make_regression(n_samples=500, n_features=4, noise=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

rf_reg = RandomForestRegressor(n_estimators=200, random_state=42, bootstrap=True, oob_score=True)

rf_reg.fit(X_train, y_train)

my_rf_reg = myRF(n_estimators=200, random_state=42, bootstrap=True, oob_score=True)

my_rf_reg.fit(X_train, y_train)

print("Sklearn:", rf_reg.predict(X_test[:5]))
print("My RF:", my_rf_reg.predict(X_test[:5]))

print("\nSklearn OOB R2:", rf_reg.oob_score_)
print("My OOB MSE:", my_rf_reg.oob_score_)

print("\nSklearn feature importances:", rf_reg.feature_importances_)
print("My feature importances:", my_rf_reg.feature_importances_)

# Sklearn: [ 50.57694837  21.19457229  25.93888229 -56.85912643  67.94352754]
# My RF: [ 53.45718943  23.57755291  23.72476057 -57.35770745  73.16915576]

# Sklearn OOB R2: 0.9079848878680892
# My OOB MSE: 0.9062953328775664

# Sklearn feature importances: [0.39354609 0.02591407 0.52505811 0.05548173]
# My feature importances: [0.39319169 0.02403908 0.52735711 0.05541212]

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from src.lib.GradientBoostingClassifier import GradientBoostClassifier as myGBC

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

gbc = GradientBoostingClassifier(max_depth=2, n_estimators=3, learning_rate=1.0)
gbc.fit(X_train, y_train)

my_gbc = myGBC(n_estimators=3, max_depth=2, learning_rate=1.0)
my_gbc.fit(X_train, y_train)

print(gbc.predict(X_test[:4]))
print(my_gbc.predict(X_test[:4]))

gbc2 = GradientBoostingClassifier(max_depth=2, n_estimators=500, learning_rate=0.05, n_iter_no_change=10, random_state=42)
gbc2.fit(X_train, y_train)
my_gbc2 = myGBC(n_estimators=500, max_depth=2, learning_rate=0.05, n_iter_no_change=10, random_state=42)
my_gbc2.fit(X_train, y_train)

gbc3 = GradientBoostingClassifier(max_depth=2, n_estimators=500, learning_rate=0.05, n_iter_no_change=10, random_state=42, subsample=0.25)
gbc3.fit(X_train, y_train)
my_gbc3 = myGBC(n_estimators=500, max_depth=2, learning_rate=0.05, n_iter_no_change=10, random_state=42, subsample=0.25)
my_gbc3.fit(X_train, y_train)

print(gbc2.predict(X_test[:4]))
print(gbc2.n_estimators_)
print(my_gbc2.predict(X_test[:4]))
print(my_gbc2.n_estimators_)

print(gbc3.predict(X_test[:4]))
print(gbc3.n_estimators_)
print(my_gbc3.predict(X_test[:4]))
print(my_gbc3.n_estimators_)

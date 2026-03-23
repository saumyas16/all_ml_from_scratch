from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from src.lib.RandomForestClassifier import RandomForestClassifier as myRF
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

rf_clf = RandomForestClassifier(n_estimators=500, random_state=42, bootstrap=True, oob_score=True, max_features='sqrt')
rf_clf.fit(X_train, y_train)

my_rf_clf = myRF(n_estimators=500, random_state=42, bootstrap=True, oob_score=True)
my_rf_clf.fit(X_train, y_train)

print(rf_clf.predict(X_test[:4]))
print(my_rf_clf.predict(X_test[:4]))

print(rf_clf.oob_score_)
print(my_rf_clf.oob_score_)

sk_y_pred = rf_clf.predict(X_test)
my_y_pred = my_rf_clf.predict(X_test)

print(accuracy_score(y_test, sk_y_pred))
print(accuracy_score(y_test, my_y_pred))

# [1 0 0 1]
# [1 0 0 1]
# 0.904
# 0.8666666666666667
# 0.888
# 0.912

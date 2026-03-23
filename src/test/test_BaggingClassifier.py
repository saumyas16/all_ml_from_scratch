from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from src.lib.BaggingClassifier import BaggingClassifier as myBagC
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, max_samples=100, n_jobs=1, random_state=42, bootstrap=True, oob_score=True)
bag_clf.fit(X_train, y_train)

my_bag_clf = myBagC(n_estimators=500, max_samples=100, random_state=42, bootstrap=True, oob_score=True)
my_bag_clf.fit(X_train, y_train)

print(bag_clf.predict(X_test[:4]))
print(my_bag_clf.predict(X_test[:4]))

print(bag_clf.oob_score_)
print(my_bag_clf.oob_score_)

sk_y_pred = bag_clf.predict(X_test)
my_y_pred = my_bag_clf.predict(X_test)

print(accuracy_score(y_test, sk_y_pred))
print(accuracy_score(y_test, my_y_pred))

# [0 0 0 1]
# [1 0 0 1]
# 0.9253333333333333
# 0.8773333333333333
# 0.904
# 0.888

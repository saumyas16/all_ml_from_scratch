from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from src.lib.AdaBoostClassifier import AdaBoostClassifier as myABC
from sklearn.metrics import accuracy_score
from src.lib.CART_decision_tree import DecisionTreeClassifier as myDTC

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=30, learning_rate=0.5)
ada_clf.fit(X_train, y_train)

my_ada_clf = myABC(classifier=myDTC(max_depth=1), learning_rate=0.5, n_estimators=30)
my_ada_clf.fit(X_train, y_train)

print(ada_clf.predict(X_test[:4]))
print(my_ada_clf.predict(X_test[:4]))

sk_y_pred = ada_clf.predict(X_test)
my_y_pred = my_ada_clf.predict(X_test)

print(accuracy_score(y_test, sk_y_pred))
print(accuracy_score(y_test, my_y_pred))

# [0 0 0 1]
# [0 0 0 1]
# 0.88
# 0.88

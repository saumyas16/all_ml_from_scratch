from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from src.lib.BaggingClassifier import BaggingClassifier as myBagC

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, max_samples=100, n_jobs=1, random_state=42)
bag_clf.fit(X_train, y_train)

my_bag_clf = myBagC(n_estimators=500, max_samples=100, random_state=42)
my_bag_clf.fit(X_train, y_train)

print(bag_clf.predict(X_test[:4]))
print(my_bag_clf.predict(X_test[:4]))

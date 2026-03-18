from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from .CART_decision_tree import DecisionTreeClassifier as MyDTC

iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y_full = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y_full, test_size=0.2, random_state=36)

tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X_train, y_train)

print(tree_clf.tree_)
print(tree_clf.predict([[5, 1.5]]))

my_tree_clf = MyDTC(max_depth=2)
my_tree_clf.fit(X_train, y_train)

print(my_tree_clf.tree_)
print(my_tree_clf.predict([[5, 1.5]]))

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from src.lib.CART_decision_tree import DecisionTreeClassifier as MyDTC

iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y_full = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y_full, test_size=0.2, random_state=36)

tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42, min_samples_split=50)
tree_clf.fit(X_train, y_train)

print("SKLearn")
print(tree_clf.predict([[5, 1.5]]))
print(tree_clf.predict_proba([[5, 1.5]]))
print(tree_clf.feature_importances_)

my_tree_clf = MyDTC(max_depth=2, min_samples_split=50)
my_tree_clf.fit(X_train, y_train)

print("MyCART")
print(my_tree_clf.tree_)
print(my_tree_clf.predict([[5, 1.5]]))
print(my_tree_clf.predict_proba([[5, 1.5]]))
print(my_tree_clf.feature_importances_)

# With Entropy

tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42, criterion="entropy")
tree_clf.fit(X_train, y_train)

print("SKLearn")
print(tree_clf.predict([[5, 1.5]]))
print(tree_clf.predict_proba([[5, 1.5]]))

my_tree_clf = MyDTC(max_depth=2, criterion="entropy")
my_tree_clf.fit(X_train, y_train)

print("MyCART")
print(my_tree_clf.tree_)
print(my_tree_clf.predict([[5, 1.5]]))
print(my_tree_clf.predict_proba([[5, 1.5]]))

# SKLearn
# [1]
# [[0.        0.9047619 0.0952381]]
# [0.57865578 0.42134422]
# MyCART
# {'feature': 0, 'threshold': 1.9, 'left': {'label': 0, 'label probability': array([1., 0., 0.]), 'gini': 0.0}, 'right': {'feature': 1, 'threshold': 1.6,
# 'left': {'label': 1, 'label probability': array([0.       , 0.9047619, 0.0952381]), 'gini': 0.172},
# 'right': {'label': 2, 'label probability': array([0.        , 0.02777778, 0.97222222]), 'gini': 0.054}}}
# [1]
# [[0.        0.9047619 0.0952381]]
# [0.57865578 0.42134422]
# SKLearn
# [1]
# [[0.        0.9047619 0.0952381]]
# MyCART
# {'feature': 0, 'threshold': 1.9, 'left': {'label': 0, 'label probability': 1.0, 'gini': 0.0}, 'right': {'feature': 1, 'threshold': 1.6,
# 'left': {'label': 1, 'label probability': array([0.       , 0.9047619, 0.0952381]), 'gini': 0.454},
# 'right': {'label': 2, 'label probability': array([0.        , 0.02777778, 0.97222222]), 'gini': 0.183}}}
# [1]
# [[0.        0.9047619 0.0952381]]

# Interesting to note that randomness affects the features importance between [0 1] to [0.57865578 0.42134422]

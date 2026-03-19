from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from src.lib.softmax_regression import LogisticRegression as MySR
from src.lib.logistic_regression import LogisticRegression as MyLR

iris = load_iris(as_frame=True)
X = iris.data[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]].values
y = (iris.target_names[iris.target] == "virginica").astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
print(log_reg.coef_)
print(log_reg.intercept_)

my_log_reg = MyLR()
my_log_reg.fit(X_train, y_train)
print(my_log_reg.coef_)

# [[-0.32823116 -0.27246843  2.69972906  2.15173888]]
# [-13.86521344]
# [-0.87763866 -1.6949895  -1.08154895  2.27032845  2.10404339]

print("Prediction of binary classes:")
print(log_reg.predict([[7.0, 3.2, 4.7, 1.4], [6.7, 3.0, 5.2, 2.3]]))
print(my_log_reg.predict([[7.0, 3.2, 4.7, 1.4], [6.7, 3.0, 5.2, 2.3]]))

y_full = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y_full, test_size=0.2, random_state=36)

log_reg2 = LogisticRegression(random_state=42)
log_reg2.fit(X_train, y_train)
print(log_reg2.coef_)
print(log_reg2.intercept_)

my_log_reg2 = MySR()
my_log_reg2.fit(X_train, y_train)
print(my_log_reg2.coef_)

print("Prediction of multiple classes:")
print(log_reg2.predict([[7.0, 3.2, 4.7, 1.4], [6.7, 3.0, 5.2, 2.3], [6.7, 3.1, 4.7, 1.5], [4.9, 2.4, 3.3, 1.0]]))
print(my_log_reg2.predict([[7.0, 3.2, 4.7, 1.4], [6.7, 3.0, 5.2, 2.3], [6.7, 3.1, 4.7, 1.5], [4.9, 2.4, 3.3, 1.0]]))

# Prediction of binary classes:
# [0 1]
# [0 1]
# [[-0.47238698  0.87702957 -2.33391381 -0.99347589]
#  [ 0.53111352 -0.43208662 -0.18504504 -0.81551852]
#  [-0.05872653 -0.44494295  2.51895884  1.8089944 ]]
# [  9.57547295   2.10134116 -11.6768141 ]
# [[ 0.91556305  0.70299765  0.26206599]
#  [ 1.35108343  1.1411225  -0.64270953]
#  [ 2.11246151  0.35435851 -0.83663599]
#  [-2.01704263  0.38871718  2.5006032 ]
#  [-0.26420469 -0.24781571  2.59219928]]
# Prediction of multiple classes:
# [1 2 1 1]
# [1 2 1 1]

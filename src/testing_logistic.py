from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from .logistic_regression import LogisticRegression as MyLR

iris = load_iris(as_frame=True)
X = iris.data[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]].values
y = iris.target_names[iris.target] == "virginica"

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

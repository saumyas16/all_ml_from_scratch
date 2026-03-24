from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from src.lib.RandomForestClassifier import RandomForestClassifier as myRF

iris = load_iris(as_frame=True)
X = iris.data[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]].values
y_full = iris.target

rf_clf = RandomForestClassifier(n_estimators=500, random_state=42)
rf_clf.fit(X, y_full)

for score, name in zip(rf_clf.feature_importances_, iris.data.columns):
    print(round(score, 3), name)

my_rf_clf = myRF(n_estimators=500, random_state=42)
my_rf_clf.fit(X, y_full)

for score, name in zip(my_rf_clf.feature_importances_, iris.data.columns):
    print(round(score, 3), name)

# 0.112 sepal length (cm)
# 0.023 sepal width (cm)
# 0.441 petal length (cm)
# 0.423 petal width (cm)
# 0.104 sepal length (cm)
# 0.007 sepal width (cm)
# 0.461 petal length (cm)
# 0.428 petal width (cm)

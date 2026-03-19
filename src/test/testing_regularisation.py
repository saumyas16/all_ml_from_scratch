import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

# Scaling is important before regularization

housing = fetch_california_housing()
scaler = StandardScaler()

X = pd.DataFrame(housing.data, columns=housing.feature_names)[["AveRooms", "MedInc", "HouseAge", "AveBedrms", "AveOccup"]]
y = housing.target
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)

ridge_reg = Ridge(alpha=0.1, solver="cholesky")
ridge_reg.fit(X_train, y_train)

print("Ridge Regression:")
print(ridge_reg.coef_)
print(ridge_reg.intercept_)

lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_train, y_train)

print("Lasso Regression:")
print(lasso_reg.coef_)
print(lasso_reg.intercept_)

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X_train, y_train)

print("ElasticNet Regression:")
print(elastic_net.coef_)
print(elastic_net.intercept_)

# Ridge Regression:
# [-0.51765551  1.01511938  0.19959641  0.47279732 -0.04525958]
# 2.0665035162781247
# Lasso Regression:
# [-0.          0.70766817  0.10470448  0.         -0.        ]
# 2.0666683678152125
# ElasticNet Regression:
# [-0.          0.7263191   0.14947095  0.         -0.        ]
# 2.066462583276885

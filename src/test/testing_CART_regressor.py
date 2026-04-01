from sklearn.tree import DecisionTreeRegressor
import numpy as np
from src.lib.CART_regressor import DecisionTreeRegressor as MyDTR

rng = np.random.default_rng(seed=42)
X_quad = rng.random((200, 1)) - 0.5
y_quad = X_quad ** 2 + 0.025 * rng.standard_normal((200, 1))

tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg.fit(X_quad, y_quad)

my_tree_reg = MyDTR(max_depth=2)
my_tree_reg.fit(X_quad, y_quad)

print(tree_reg.predict([[0.038]]))
print(my_tree_reg.predict([[0.038]]))

# [0.03758823]
# [0.03758823]

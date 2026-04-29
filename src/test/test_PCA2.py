from src.lib.PCA import PCA as MyPCA
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error

mnist = fetch_openml('mnist_784', as_frame=False)

X_train, y_train = mnist.data[:6_000], mnist.target[:6_000]
X_test, y_test = mnist.data[6_000:], mnist.target[6_000:]

pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train)
print(pca.n_components_)

mypca = MyPCA(n_components=0.95)
X_reduced2 = mypca.fit_transform(X_train)
print(mypca.n_components)

X_recovered = pca.inverse_transform(X_reduced)
X_recovered2 = mypca.inverse_transform(X_reduced2)

print("Reconstruction error sklearn PCA:", mean_squared_error(X_train, X_recovered))
print("Reconstruction error my PCA:", mean_squared_error(X_train, X_recovered2))

# 149
# 149
# Reconstruction error sklearn PCA: 217.40487501345987
# Reconstruction error my PCA: 217.40487501345987

import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.components_ = None

    def fit_transform(self, X):
        X = np.asarray(X)
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        U, s, Vt = np.linalg.svd(X_centered)

        variance_ratio_ = s ** 2 / np.sum(s ** 2)

        if self.n_components > 0.0 and self.n_components < 1.0:
            variance_sum = np.cumsum(variance_ratio_)
            d = np.argmax(variance_sum >= self.n_components) + 1
            self.n_components = d

        self.components_ = Vt[:self.n_components]
        X_d = X_centered @ self.components_.T
        self.explained_variance_ratio_ = variance_ratio_[:self.n_components]

        return X_d

    def inverse_transform(self, X):
        X = np.asarray(X)
        return X @ self.components_ + self.mean_

import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.explained_variance_ratio_ = None

    def fit_transform(self, X):
        X = np.asarray(X)
        X_centered = X - X.mean(axis=0)
        U, s, Vt = np.linalg.svd(X_centered)

        variance_ratio_ = s ** 2 / np.sum(s ** 2)

        W_d = Vt[:self.n_components].T
        X_d = X_centered @ W_d
        self.explained_variance_ratio_ = variance_ratio_[:self.n_components]

        return X_d

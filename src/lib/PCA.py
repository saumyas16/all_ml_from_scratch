import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        X = np.asarray(X)
        X_centered = X - X.mean(axis=0)
        U, s, Vt = np.linalg.svd(X_centered)

        W_d = Vt[:self.n_components].T
        X_d = X_centered @ W_d

        return X_d

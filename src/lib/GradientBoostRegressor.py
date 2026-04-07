import numpy as np
import random
from src.lib.CART_regressor import DecisionTreeRegressor


class GradientBoostRegressor:
    def __init__(self, n_estimators, max_depth, min_samples_split=2, n_iter_no_change=None, max_features=1.0, subsample=1, random_state=None,
                 learning_rate=1.0, validation_fraction=0.1, tol=0.0001):
        self.n_estimators = n_estimators
        self.n_iter_no_change = n_iter_no_change
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.subsample = subsample
        self.rng = random.Random(random_state) or random
        self.learning_rate = learning_rate
        self.validation_fraction = validation_fraction
        self.tol = tol
        self.predictors_info = None
        self.init_ = None

    @staticmethod
    def _train_regressor(regressor, X, y, max_depth, min_samples_split, max_features):
        reg = regressor(max_depth=max_depth, min_samples_split=min_samples_split, max_features=max_features)
        reg.fit(X, y)
        return reg

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.init_ = np.mean(y)   # works the same with 0 initialization for simplest case
        residual = y - self.init_
        predictors = []
        for i in range(self.n_estimators):
            i_reg = GradientBoostRegressor._train_regressor(DecisionTreeRegressor, X, residual, self.max_depth, self.min_samples_split, self.max_features)
            y_pred = i_reg.predict(X)
            residual -= self.learning_rate * y_pred
            predictors.append(i_reg)

        self.predictors_info = predictors

    def _predict_one(self, x):
        x = np.asarray(x).reshape(1, -1)
        pred = self.init_
        for reg in self.predictors_info:
            pred += self.learning_rate * reg.predict(x)[0]

        return pred

    def predict(self, X):
        predictions = []
        for x in X:
            pred = self._predict_one(x)
            predictions.append(pred)

        return np.array(predictions)

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
        self.n_estimators_ = 0

    @staticmethod
    def _train_regressor(regressor, X, y, max_depth, min_samples_split, max_features):
        reg = regressor(max_depth=max_depth, min_samples_split=min_samples_split, max_features=max_features)
        reg.fit(X, y)
        return reg

    def squared_loss(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        numExamples, numFeatures = X.shape
        val_X = None
        val_y = None
        cnt = None
        best_loss = None
        pred_val = None

        if self.n_iter_no_change is not None:
            validation_samples = int(self.validation_fraction * numExamples)
            val_idxs = self.rng.sample(range(0, numExamples), k=validation_samples)
            train_idxs = [i for i in range(0, numExamples) if i not in val_idxs]
            val_X = X[val_idxs, :]
            val_y = y[val_idxs]
            X = X[train_idxs, :]
            y = y[train_idxs]
            cnt = 0
            best_loss = float("inf")

        self.init_ = np.mean(y)   # works the same with 0 initialization for simplest case
        residual = y - self.init_
        if val_X is not None:
            pred_val = self.init_
        predictors = []
        for i in range(self.n_estimators):
            i_reg = GradientBoostRegressor._train_regressor(DecisionTreeRegressor, X, residual, self.max_depth, self.min_samples_split, self.max_features)
            y_pred = i_reg.predict(X)
            residual -= self.learning_rate * y_pred
            predictors.append(i_reg)
            if cnt is not None:
                y_val_pred = i_reg.predict(val_X)
                pred_val += self.learning_rate * y_val_pred
                curr_loss = GradientBoostRegressor.squared_loss(val_y, pred_val)
                if (best_loss - curr_loss) <= self.tol:
                    cnt += 1
                else:
                    cnt = 0
                    best_loss = curr_loss

                if cnt >= self.n_iter_no_change:
                    self.n_estimators_ = i
                    break
        if self.n_estimators_ == 0:
            self.n_estimators_ = self.n_estimators
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

import numpy as np
import random
from src.lib.DecisionTreeRegressor import DecisionTreeRegressor


class GradientBoostClassifier:
    def __init__(self, n_estimators, max_depth, min_samples_split=2, n_iter_no_change=None, max_features=1.0, subsample=1.0, random_state=None,
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
        self.n_classes_ = None

    @staticmethod
    def _train_regressor(regressor, X, y, max_depth, min_samples_split, max_features):
        reg = regressor(max_depth=max_depth, min_samples_split=min_samples_split, max_features=max_features)
        reg.fit(X, y)
        return reg

    def log_loss(y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))

    def sigmoid(t):
        return 1/(1+np.exp(-t))

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        numExamples, numFeatures = X.shape
        self.n_classes_ = np.unique(y).size
        val_X = None
        val_y = None
        cnt = None
        best_loss = None
        F_val = None

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

        p = np.clip(np.mean(y), 1e-15, 1 - 1e-15)
        self.init_ = np.log(p / (1 - p))

        if val_X is not None:
            F_val = np.full(val_y.shape, self.init_)
        predictors = []
        F = np.full(y.shape, self.init_)
        p = GradientBoostClassifier.sigmoid(F)
        residual = y - p

        for i in range(self.n_estimators):
            numSamples = max(1, int(self.subsample * y.size))
            train_idxs = self.rng.sample(range(0, y.size), k=numSamples)
            X_train = X[train_idxs, :]
            residual_train = residual[train_idxs]

            i_reg = GradientBoostClassifier._train_regressor(DecisionTreeRegressor, X_train, residual_train, self.max_depth, self.min_samples_split, self.max_features)
            # y_pred = i_reg.predict(X)
            # F += self.learning_rate * y_pred
            leaf_idx = i_reg.apply(X)
            leaf_values = {}
            for leaf in np.unique(leaf_idx):
                idx = leaf_idx == leaf
                gamma = sum(residual[idx]) / sum(p[idx] * (1 - p[idx]) + 1e-10)
                leaf_values[leaf] = gamma
                F[idx] += self.learning_rate * gamma
            p = GradientBoostClassifier.sigmoid(F)
            residual = y - p
            predictors.append((i_reg, leaf_values))

            if cnt is not None:
                # y_val_pred = i_reg.predict(val_X)
                # F_val += self.learning_rate * y_val_pred
                leaf_idx_val = i_reg.apply(val_X)
                for leaf in np.unique(leaf_idx_val):
                    idx = leaf_idx_val == leaf
                    F_val[idx] += self.learning_rate * leaf_values[leaf]
                p_val = GradientBoostClassifier.sigmoid(F_val)
                curr_loss = GradientBoostClassifier.log_loss(val_y, p_val)
                if (best_loss - curr_loss) <= self.tol:
                    cnt += 1
                else:
                    cnt = 0
                    best_loss = curr_loss

                if cnt >= self.n_iter_no_change:
                    self.n_estimators_ = i + 1
                    break
        if self.n_estimators_ == 0:
            self.n_estimators_ = self.n_estimators
        self.predictors_info = predictors

    def _predict_one(self, x):
        x = np.asarray(x).reshape(1, -1)
        pred = self.init_
        for reg, leaf_values in self.predictors_info:
            leaf = reg.apply(x)[0]
            pred += self.learning_rate * leaf_values[leaf]

        prob = GradientBoostClassifier.sigmoid(pred)
        return (prob >= 0.5).astype(int)

    def predict(self, X):
        predictions = []
        for x in X:
            pred = self._predict_one(x)
            predictions.append(pred)

        return np.array(predictions)

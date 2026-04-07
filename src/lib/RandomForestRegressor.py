import numpy as np
import random
from src.lib.CART_regressor import DecisionTreeRegressor


class RandomForestRegressor:
    def __init__(self, n_estimators, random_state=None, max_features=1.0, max_samples=1, max_depth=None, min_samples_split=2,
                 bootstrap=True, oob_score=False, bootstrap_features=False):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.trees_info = None
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score_ = None
        self.calculate_oob = oob_score
        self.rng = random.Random(random_state) or random
        self.feature_importances_ = None

    def sampling(self, X, y, max_samples=None, bootstrap_samples=True):
        numExamples, numFeatures = X.shape
        bag_idxs = None

        if max_samples is None:
            max_samples = numExamples

        if bootstrap_samples:
            bag_idxs = self.rng.choices(range(0, numExamples), k=max_samples)
        else:
            max_samples = min(max_samples, numExamples)
            bag_idxs = self.rng.sample(range(0, numExamples), k=max_samples)

        bag_set = set(bag_idxs)
        oob_idxs = [i for i in range(0, numExamples) if i not in bag_set]

        X_b, y_b = X[bag_idxs, :], y[bag_idxs]

        return X_b, y_b, np.asarray(oob_idxs)

    @staticmethod
    def _train_regressor(X, y, max_depth, min_samples_split, max_features, regressor):
        tree_reg = regressor(max_depth=max_depth, min_samples_split=min_samples_split, max_features=max_features)
        tree_reg.fit(X, y)
        return tree_reg

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        numExamples, numFeatures = X.shape
        if self.max_samples > 0 and self.max_samples <= 1:
            self.max_samples *= numExamples
            self.max_samples = int(self.max_samples)

        if self.max_features > 0 and self.max_features <= 1:
            self.max_features *= numFeatures
            self.max_features = max(1, int(self.max_features))

        trees = []
        oob_idx_trees = [[] for i in range(numExamples)]
        self.feature_importances_ = np.zeros(numFeatures)

        for i in range(self.n_estimators):
            X_sample, y_sample, oob_idxs = RandomForestRegressor.sampling(self, X, y, self.max_samples, self.bootstrap)
            i_tree = RandomForestRegressor._train_regressor(X_sample, y_sample, self.max_depth, self.min_samples_split, self.max_features, DecisionTreeRegressor)
            trees.append(i_tree)
            for idx in oob_idxs:
                oob_idx_trees[idx].append(i_tree)
            self.feature_importances_ += i_tree._rf_feature_importance

        self.feature_importances_ /= np.sum(self.feature_importances_)

        if self.calculate_oob:
            valid_cnt = np.zeros(numExamples)
            pred_y = np.zeros(numExamples)
            for i in range(numExamples):
                x = X[i].reshape(1, -1)
                if len(oob_idx_trees[i]) == 0:
                    continue
                sum_i = 0
                cnt = 0
                for tree in oob_idx_trees[i]:
                    pred_i = tree.predict(x)
                    sum_i += pred_i[0]
                    cnt += 1

                pred_y[i] = sum_i/cnt
                valid_cnt[i] += 1

            mask = valid_cnt > 0
            y_true = y[mask]
            y_pred = pred_y[mask]

            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

            if ss_tot == 0:
                self.oob_score_ = 0.0
            else:
                self.oob_score_ = 1 - (ss_res / ss_tot)

        self.trees_info = trees

    def _predict_one(self, x):
        sum_i = 0
        cnt = 0

        x = np.asarray(x)
        if x.ndim == 0:
            x = x.reshape(1, 1)
        elif x.ndim == 1:
            x = x.reshape(1, -1)

        for tree in self.trees_info:
            prediction_i = tree.predict(x)
            sum_i += prediction_i[0]
            cnt += 1

        return sum_i/cnt

    def predict(self, X):
        predictions = []
        for x in X:
            pred = self._predict_one(x)
            predictions.append(pred)

        return np.array(predictions)

import numpy as np
import random

np.set_printoptions(legacy='1.25')


class DecisionTreeRegressor:
    def __init__(self, max_features=1.0, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.split_info = {}
        self.tree_ = None
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.feature_importances_ = None
        self._rf_feature_importance = None
        self.n_samples = None
        self.total_weight = None
        self.leaf_id = -1

    @staticmethod
    def mse_measure(node_y, w):
        y_mean = np.sum(w * node_y) / np.sum(w)
        mse = np.sum(w * (node_y - y_mean)**2) / np.sum(w)
        return mse

    @staticmethod
    def node_val(node_y, w):
        return np.sum(w * node_y) / np.sum(w)

    @staticmethod
    def cost_function(node_X, node_y, feature_idx, feature_threshold, weights):
        node_left = node_y[node_X[:, feature_idx] <= feature_threshold]
        node_right = node_y[node_X[:, feature_idx] > feature_threshold]
        weight_left = weights[node_X[:, feature_idx] <= feature_threshold]
        weight_right = weights[node_X[:, feature_idx] > feature_threshold]

        if node_left.size == 0 or node_right.size == 0:
            return float("inf")

        mse_left = DecisionTreeRegressor.mse_measure(node_left, weight_left)
        mse_right = DecisionTreeRegressor.mse_measure(node_right, weight_right)

        cost_value = (np.sum(weight_left) * mse_left + np.sum(weight_right) * mse_right) / np.sum(weights)

        return cost_value

    @staticmethod
    def threshold_list(X):
        X = np.sort(X)
        threshold = []
        for i in range(0, X.size - 1):
            threshold.append((X[i]+X[i+1])/2)
        return np.unique(threshold)

    def split_examples(self, X, y, sample_weights, numFeatures, curr_depth):
        node_value = DecisionTreeRegressor.node_val(y, sample_weights)
        node_mse = DecisionTreeRegressor.mse_measure(y, sample_weights)
        if self.max_depth is not None and curr_depth >= self.max_depth:
            self.leaf_id += 1
            return {"value": node_value, "mse": node_mse, "leaf_id": self.leaf_id}
        elif y.size < self.min_samples_split:
            self.leaf_id += 1
            return {"value": node_value, "mse": node_mse, "leaf_id": self.leaf_id}
        elif np.unique(y).size == 1:
            self.leaf_id += 1
            return {"value": y[0], "mse": 0, "leaf_id": self.leaf_id}
        else:
            best_cost = float("inf")
            splitinfo = {}

            feature_subset = random.sample(range(numFeatures), self.max_features)
            for p in feature_subset:
                thresholds = DecisionTreeRegressor.threshold_list(X[:, p])
                for feature_threshold in thresholds:
                    i_cost = DecisionTreeRegressor.cost_function(X, y, p, feature_threshold, sample_weights)
                    if i_cost < best_cost:
                        best_cost = i_cost
                        splitinfo = {"feature": p, "threshold": feature_threshold, "left": None, "right": None}

            if best_cost >= node_mse:
                self.leaf_id += 1
                return {"value": node_value, "mse": node_mse, "leaf_id": self.leaf_id}
            if not splitinfo:
                self.leaf_id += 1
                return {"value": node_value, "mse": node_mse, "leaf_id": self.leaf_id}

            self.feature_importances_[splitinfo["feature"]] += (node_mse-best_cost)*(np.sum(sample_weights) / self.total_weight)
            self._rf_feature_importance[splitinfo["feature"]] += (node_mse-best_cost)*(y.size)

            decision_boundary = X[:, splitinfo["feature"]] <= splitinfo["threshold"]
            splitinfo["left"] = DecisionTreeRegressor.split_examples(self, X[decision_boundary], y[decision_boundary], sample_weights[decision_boundary], numFeatures, curr_depth+1)
            splitinfo["right"] = DecisionTreeRegressor.split_examples(self, X[~decision_boundary], y[~decision_boundary], sample_weights[~decision_boundary], numFeatures, curr_depth+1)

            return splitinfo

    def fit(self, X, y, sample_weights=None):
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        if sample_weights is None:
            sample_weights = np.ones(len(y))
        else:
            sample_weights = np.asarray(sample_weights)
        if X.ndim == 0:
            X = X.reshape(1, 1)
        elif X.ndim == 1:
            X = X.reshape(-1, 1)
        numExamples, numFeatures = X.shape
        self.n_samples = numExamples
        self.feature_importances_ = np.zeros(numFeatures)
        self._rf_feature_importance = np.zeros(numFeatures)
        self.total_weight = np.sum(sample_weights)

        if self.max_features > 0 and self.max_features <= 1:
            self.max_features *= numFeatures
            self.max_features = max(1, int(self.max_features))

        self.leaf_id = -1
        self.tree_ = DecisionTreeRegressor.split_examples(self, X, y, sample_weights, numFeatures, 0)

        self.feature_importances_ /= np.sum(self.feature_importances_)

    def _predict_one(self, x, node):
        if "value" in node:
            return node["value"]
        else:
            feature = node["feature"]
            threshold = node["threshold"]

            if x[feature] <= threshold:
                return self._predict_one(x, node["left"])
            else:
                return self._predict_one(x, node["right"])

    def predict(self, X):
        predictions = []
        for x in X:
            pred = self._predict_one(x, self.tree_)
            predictions.append(pred)

        return np.array(predictions)

    def _apply_one(self, x, node):
        if "value" in node:
            return node["leaf_id"]
        else:
            feature = node["feature"]
            threshold = node["threshold"]

            if x[feature] <= threshold:
                return self._apply_one(x, node["left"])
            else:
                return self._apply_one(x, node["right"])

    def apply(self, X):
        leaf_indices = []
        for x in X:
            idx = self._apply_one(x, self.tree_)
            leaf_indices.append(idx)

        return np.array(leaf_indices)

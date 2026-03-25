import numpy as np
import random

np.set_printoptions(legacy='1.25')


class DecisionTreeRegressor:
    def __init__(self, max_features=1, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.split_info = {}
        self.tree_ = None
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.feature_importances_ = None
        self._rf_feature_importance = None
        self.n_samples = None

    @staticmethod
    def mse_measure(node_y):
        y_mean = np.mean(node_y)
        mse = np.mean((y_mean - node_y)**2)
        return mse

    @staticmethod
    def node_val(node_y):
        return np.mean(node_y)

    @staticmethod
    def cost_function(node_X, node_y, feature_idx, feature_threshold):
        node_left = node_y[node_X[:, feature_idx] <= feature_threshold]
        node_right = node_y[node_X[:, feature_idx] > feature_threshold]

        if node_left.size == 0 or node_right.size == 0:
            return float("inf")

        mse_left = DecisionTreeRegressor.mse_measure(node_left)
        mse_right = DecisionTreeRegressor.mse_measure(node_right)

        cost_value = ((node_left.size)*mse_left + (node_right.size)*mse_right)/node_y.size
        return cost_value

    @staticmethod
    def threshold_list(X):
        X = np.sort(X)
        threshold = []
        for i in range(0, X.size - 1):
            threshold.append((X[i]+X[i+1])/2)
        return np.unique(threshold)

    def split_examples(self, X, y, numFeatures, curr_depth):
        node_value = DecisionTreeRegressor.node_val(y)
        node_mse = DecisionTreeRegressor.mse_measure(y)
        if self.max_depth is not None and curr_depth >= self.max_depth:
            return {"value": node_value, "mse": node_mse}
        elif y.size < self.min_samples_split:
            return {"value": node_value, "mse": node_mse}
        elif np.unique(y).size == 1:
            return {"value": y[0], "mse": 0}
        else:
            best_cost = float("inf")
            splitinfo = {}

            feature_subset = random.sample(range(numFeatures), self.max_features)
            for p in feature_subset:
                thresholds = DecisionTreeRegressor.threshold_list(X[:, p])
                for feature_threshold in thresholds:
                    i_cost = DecisionTreeRegressor.cost_function(X, y, p, feature_threshold)
                    if i_cost < best_cost:
                        best_cost = i_cost
                        splitinfo = {"feature": p, "threshold": feature_threshold, "left": None, "right": None}

            if best_cost >= node_mse:
                return {"value": node_value, "mse": node_mse}
            if not splitinfo:
                return {"value": node_value, "mse": node_mse}

            self.feature_importances_[splitinfo["feature"]] += (node_mse-best_cost)*(y.size / self.n_samples)
            self._rf_feature_importance[splitinfo["feature"]] += (node_mse-best_cost)*(y.size)

            decision_boundary = X[:, splitinfo["feature"]] <= splitinfo["threshold"]
            splitinfo["left"] = DecisionTreeRegressor.split_examples(self, X[decision_boundary], y[decision_boundary], numFeatures, curr_depth+1)
            splitinfo["right"] = DecisionTreeRegressor.split_examples(self, X[~decision_boundary], y[~decision_boundary], numFeatures, curr_depth+1)

            return splitinfo

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim == 0:
            X = X.reshape(1, 1)
        elif X.ndim == 1:
            X = X.reshape(-1, 1)
        numExamples, numFeatures = X.shape
        self.n_samples = numExamples
        self.feature_importances_ = np.zeros(numFeatures)
        self._rf_feature_importance = np.zeros(numFeatures)

        if self.max_features > 0 and self.max_features <= 1:
            self.max_features *= numFeatures
            self.max_features = max(1, int(self.max_features))

        self.tree_ = DecisionTreeRegressor.split_examples(self, X, y, numFeatures, 0)

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

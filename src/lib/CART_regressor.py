import numpy as np


class DecisionTreeRegressor:
    def __init__(self, max_depth=2, min_samples_split=2):
        self.coef_ = None
        self.accuracy_ = None
        self.max_depth = max_depth
        self.split_info = {}
        self.tree_ = None
        self.min_samples_split = min_samples_split

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
            return 1

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

    def split_examples(self, max_depth, X, y, numFeatures, prev_cost, depth, min_split_samples):
        node_value = DecisionTreeRegressor.node_val(y)
        if max_depth is not None and depth >= max_depth:
            return {"value": node_value}
        elif np.unique(y).size == 1:
            return {"value": y[0]}
        else:
            best_cost = float("inf")
            splitinfo = {}

            for p in range(0, numFeatures):
                thresholds = DecisionTreeRegressor.threshold_list(X[:, p])
                for feature_threshold in thresholds:
                    i_cost = DecisionTreeRegressor.cost_function(X, y, p, feature_threshold)
                    if i_cost < best_cost:
                        best_cost = i_cost
                        splitinfo = {"feature": p, "threshold": feature_threshold, "left": None, "right": None}

            if best_cost >= prev_cost:
                return {"value": node_value}
            if not splitinfo or X.size < min_split_samples:
                return {"value": node_value}
            decision_boundary = X[:, splitinfo["feature"]] <= splitinfo["threshold"]
            splitinfo["left"] = DecisionTreeRegressor.split_examples(self, max_depth, X[decision_boundary], y[decision_boundary], numFeatures, best_cost, depth+1, min_split_samples)
            splitinfo["right"] = DecisionTreeRegressor.split_examples(self, max_depth, X[~decision_boundary], y[~decision_boundary], numFeatures, best_cost, depth+1, min_split_samples)

            return splitinfo

    def fit(self, X, y):
        if X.ndim == 0:
            X = X.reshape(1, 1)
        elif X.ndim == 1:
            X = X.reshape(-1, 1)
        numExamples, numFeatures = X.shape

        root_cost = DecisionTreeRegressor.mse_measure(y)  # root cost
        self.tree_ = DecisionTreeRegressor.split_examples(self, self.max_depth, X, y, numFeatures, root_cost, 0, self.min_samples_split)

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

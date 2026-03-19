import numpy as np
import math


class DecisionTreeClassifier:
    def __init__(self, max_depth=2, criterion="gini", min_samples_split=2):
        self.coef_ = None
        self.accuracy_ = None
        self.max_depth = max_depth
        self.split_info = {}
        self.tree_ = None
        self.criterion = criterion
        self.min_samples_split = min_samples_split

    @staticmethod
    def impurity_measure(node_y, k, criterion):
        node_size = node_y.size
        if criterion == "gini":
            gini = 1.0
            for i in range(k):
                p_i = np.sum(node_y == i) / node_size
                gini -= p_i ** 2
            return gini
        else:
            entropy = 0.0
            for i in range(k):
                p_i = np.sum(node_y == i) / node_size
                if p_i > 0:
                    entropy -= p_i * math.log2(p_i)
            return entropy

    @staticmethod
    def node_label(node_y, k):
        max_k = 0
        max_count = 0
        for i in range(k):
            class_count = np.sum(node_y == i)
            if class_count > max_count:
                max_count = class_count
                max_k = i
        return max_k

    @staticmethod
    def cost_function(node_X, node_y, k, feature_idx, feature_threshold, criterion):
        node_left = node_y[node_X[:, feature_idx] <= feature_threshold]
        node_right = node_y[node_X[:, feature_idx] > feature_threshold]

        if node_left.size == 0 or node_right.size == 0:
            return 1

        impurity_left = DecisionTreeClassifier.impurity_measure(node_left, k, criterion)
        impurity_right = DecisionTreeClassifier.impurity_measure(node_right, k, criterion)

        cost_value = ((node_left.size)*impurity_left + (node_right.size)*impurity_right)/node_y.size
        return cost_value

    @staticmethod
    def threshold_list(X):
        X = np.sort(X)
        threshold = []
        for i in range(0, X.size - 1):
            threshold.append((X[i]+X[i+1])/2)
        return np.unique(threshold)

    def split_examples(self, max_depth, X, y, k, numFeatures, prev_cost, depth, min_split_samples):
        label_node = DecisionTreeClassifier.node_label(y, k)
        if max_depth is not None and depth >= max_depth:
            return {"label": label_node}
        elif np.unique(y).size == 1:
            return {"label": y[0]}
        else:
            # best_cost = 1.0 # works for gini but not entropy
            best_cost = float("inf")
            splitinfo = {}

            for p in range(0, numFeatures):
                thresholds = DecisionTreeClassifier.threshold_list(X[:, p])
                for feature_threshold in thresholds:
                    i_cost = DecisionTreeClassifier.cost_function(X, y, k, p, feature_threshold, self.criterion)
                    if i_cost < best_cost:
                        best_cost = i_cost
                        splitinfo = {"feature": p, "threshold": feature_threshold, "left": None, "right": None}

            if best_cost >= prev_cost:
                return {"label": label_node}
            if not splitinfo or X.size < min_split_samples:
                return {"label": label_node}
            decision_boundary = X[:, splitinfo["feature"]] <= splitinfo["threshold"]
            splitinfo["left"] = DecisionTreeClassifier.split_examples(self, max_depth, X[decision_boundary], y[decision_boundary], k, numFeatures, best_cost, depth+1, min_split_samples)
            splitinfo["right"] = DecisionTreeClassifier.split_examples(self, max_depth, X[~decision_boundary], y[~decision_boundary], k, numFeatures, best_cost, depth+1, min_split_samples)

            return splitinfo

    def fit(self, X, y):
        numExamples, numFeatures = X.shape
        k = np.unique(y).size

        root_cost = DecisionTreeClassifier.impurity_measure(y, k, self.criterion)  # root cost to handle entropy, gini could be capped at 1
        self.tree_ = DecisionTreeClassifier.split_examples(self, self.max_depth, X, y, k, numFeatures, root_cost, 0, self.min_samples_split)

    def _predict_one(self, x, node):
        if "label" in node:
            return node["label"]
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

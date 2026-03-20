import numpy as np
import random
from src.lib.CART_decision_tree import DecisionTreeClassifier


class BaggingClassifier:
    def __init__(self, n_estimators, max_samples, random_state, classifier=DecisionTreeClassifier, max_depth=2, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.classifier = classifier
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees_info = None
        self.n_classes_ = None
        random.seed(random_state)

    @staticmethod
    def bagging_samples(max_samples, X, y):
        numExamples, numFeatures = X.shape
        bag_idxs = random.choices(range(0, numExamples), k=max_samples)
        X_b, y_b = X[bag_idxs, :], y[bag_idxs]

        return X_b, y_b

    @staticmethod
    def _train_classifier(X, y, max_depth, min_samples_split, classifier):
        tree_clf = classifier(max_depth=max_depth, min_samples_split=min_samples_split)
        tree_clf.fit(X, y)
        return tree_clf

    def fit(self, X, y):
        numExamples, numFeatures = X.shape
        if self.max_samples > 0 and self.max_samples <= 1:
            self.max_samples *= numExamples
            self.max_samples = int(self.max_samples)

        self.n_classes_ = np.unique(y).size
        trees = []
        for i in range(self.n_estimators):
            X_sample, y_sample = BaggingClassifier.bagging_samples(self.max_samples, X, y)
            i_tree = BaggingClassifier._train_classifier(X_sample, y_sample, self.max_depth, self.min_samples_split, self.classifier)
            trees.append(i_tree)

        self.trees_info = trees

    def _predict_one(self, x):
        votes = np.zeros((self.n_classes_))

        x = np.asarray(x)
        if x.ndim == 0:
            x = x.reshape(1, 1)
        elif x.ndim == 1:
            x = x.reshape(1, -1)

        for tree in self.trees_info:
            prediction_i = tree.predict(x)
            votes[prediction_i[0]] += 1

        return np.argmax(votes)

    def predict(self, X):
        predictions = []
        for x in X:
            pred = self._predict_one(x)
            predictions.append(pred)

        return np.array(predictions)

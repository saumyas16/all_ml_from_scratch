import numpy as np
import random
from src.lib.CART_decision_tree import DecisionTreeClassifier


class BaggingClassifier:
    def __init__(self, n_estimators, max_samples, max_features, random_state, classifier=DecisionTreeClassifier, max_depth=2, min_samples_split=2, voting="soft",
                 bootstrap=True, oob_score=False, bootstrap_features=False):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.classifier = classifier
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.trees_info = None
        self.n_classes_ = None
        self.voting_type = voting
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score_ = None
        self.calculate_oob = oob_score
        random.seed(random_state)

    @staticmethod
    def sampling(X, y, max_samples=None, bootstrap_samples=True, max_features=None, bootstrap_feature=False):
        numExamples, numFeatures = X.shape
        bag_idxs = None
        feature_idxs = list(range(0, numFeatures))

        if max_samples is None:
            max_samples = numExamples

        if bootstrap_samples:
            bag_idxs = random.choices(range(0, numExamples), k=max_samples)
        else:
            max_samples = min(max_samples, numExamples)
            bag_idxs = random.sample(range(0, numExamples), k=max_samples)

        bag_set = set(bag_idxs)
        oob_idxs = [i for i in range(0, numExamples) if i not in bag_set]

        if max_features is None:
            max_features = numFeatures

        if bootstrap_feature:
            feature_idxs = random.choices(range(0, numFeatures), k=max_features)
        else:
            max_features = min(max_features, numFeatures)
            feature_idxs = random.sample(range(0, numFeatures), k=max_features)

        X_b, y_b = X[bag_idxs, feature_idxs], y[bag_idxs]

        return X_b, y_b, np.asarray(oob_idxs), feature_idxs

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

        if self.max_features > 0 and self.max_features <= 1:
            self.max_features *= numFeatures
            self.max_features = max(1, int(self.max_features))

        self.n_classes_ = np.unique(y).size
        trees = []
        oob_idx_trees = [[] for i in range(numExamples)]
        for i in range(self.n_estimators):
            X_sample, y_sample, oob_idxs, feature_idxs = BaggingClassifier.sampling(X, y, self.max_samples, self.bootstrap, self.max_features, self.bootstrap_features)
            i_tree = BaggingClassifier._train_classifier(X_sample, y_sample, self.max_depth, self.min_samples_split, self.classifier)
            trees.append((i_tree, feature_idxs))
            for idx in oob_idxs:
                oob_idx_trees[idx].append((i_tree, feature_idxs))

        if self.calculate_oob:
            correct_cnt = 0
            valid_cnt = 0
            for i in range(numExamples):
                x = X[i].reshape(1, -1)
                if len(oob_idx_trees[i]) == 0:
                    continue
                if self.voting_type == "hard":
                    votes = np.zeros((self.n_classes_))
                    for tree, f_idxs in oob_idx_trees[i]:
                        pred_i = tree.predict(x[:, f_idxs])
                        votes[pred_i[0]] += 1
                else:
                    added_proba = np.zeros((1, self.n_classes_))
                    for tree, f_idxs in oob_idx_trees[i]:
                        pred_i = tree.predict_proba(x[:, f_idxs])
                        added_proba += pred_i
                    votes = added_proba[0]/len(oob_idx_trees[i])

                pred = np.argmax(votes)
                valid_cnt += 1
                if pred == y[i]:
                    correct_cnt += 1

            self.oob_score_ = correct_cnt/valid_cnt

        self.trees_info = trees

    def _predict_one(self, x):
        votes = np.zeros((self.n_classes_))

        x = np.asarray(x)
        if x.ndim == 0:
            x = x.reshape(1, 1)
        elif x.ndim == 1:
            x = x.reshape(1, -1)

        if self.voting_type == "hard":
            for tree, f_idxs in self.trees_info:
                prediction_i = tree.predict(x[:, f_idxs])
                votes[prediction_i[0]] += 1
        else:
            added_proba = np.zeros((1, self.n_classes_))
            for tree, f_idxs in self.trees_info:
                prediction_i = tree.predict_proba(x[:, f_idxs])
                added_proba += prediction_i
            votes = added_proba[0]/self.n_estimators

        return np.argmax(votes)

    def predict(self, X):
        predictions = []
        for x in X:
            pred = self._predict_one(x)
            predictions.append(pred)

        return np.array(predictions)

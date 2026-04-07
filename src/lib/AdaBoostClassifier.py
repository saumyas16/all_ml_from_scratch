import numpy as np
import math
import copy


class AdaBoostClassifier:
    def __init__(self, n_estimators, classifier, random_state=None, learning_rate=1, voting_type="hard"):
        self.learning_rate = learning_rate
        self.classifier = classifier
        self.n_estimators = n_estimators
        self.n_classes_ = None
        self.predictors_info = None
        self.voting_type = voting_type

    @staticmethod
    def weighted_error_rate(predicted_y, y, weights_examples):
        error_rate = 0
        for i in range(len(weights_examples)):
            if predicted_y[i] != y[i]:
                error_rate += weights_examples[i]
        return error_rate

    @staticmethod
    def predictor_weight(predicted_y, y, weights_examples, learning_rate, k):
        r = AdaBoostClassifier.weighted_error_rate(predicted_y, y, weights_examples)
        eps = 1e-10
        r = np.clip(r, eps, 1 - eps)
        alpha_j = learning_rate*(math.log((1-r)/r) + math.log(k - 1))
        return alpha_j

    @staticmethod
    def update_weights(weights_examples, y, predicted_y, weight_i):
        sum_w = 0
        for i in range(len(weights_examples)):
            if predicted_y[i] != y[i]:
                weights_examples[i] *= math.exp(weight_i)
            sum_w += weights_examples[i]
        weights_examples /= sum_w
        return weights_examples

    @staticmethod
    def _train_classifier(X, y, classifier, weights_examples):
        tree_clf = copy.deepcopy(classifier)
        tree_clf.fit(X, y, sample_weights=weights_examples)
        return tree_clf

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        numExamples, numFeatures = X.shape
        self.n_classes_ = np.unique(y).size

        weights_examples = np.ones(numExamples)
        weights_examples /= numExamples

        predictors = []

        for i in range(self.n_estimators):
            i_predictor = AdaBoostClassifier._train_classifier(X, y, self.classifier, weights_examples)
            predicted_y = i_predictor.predict(X)
            weight_i = AdaBoostClassifier.predictor_weight(predicted_y, y, weights_examples, self.learning_rate, self.n_classes_)
            predictors.append((i_predictor, weight_i))
            weights_examples = AdaBoostClassifier.update_weights(weights_examples, y, predicted_y, weight_i)

        self.predictors_info = predictors

    def _predict_one(self, x):
        votes = np.zeros((self.n_classes_))

        x = np.asarray(x)
        if x.ndim == 0:
            x = x.reshape(1, 1)
        elif x.ndim == 1:
            x = x.reshape(1, -1)

        if self.voting_type == "hard":
            for predictor, weight in self.predictors_info:
                prediction_i = predictor.predict(x)
                votes[prediction_i[0]] += weight
        else:
            added_proba = np.zeros((1, self.n_classes_))
            for predictor, weight in self.predictors_info:
                prediction_i = predictor.predict_proba(x)
                added_proba += weight*prediction_i
            votes = added_proba[0]/np.sum([w for _, w in self.predictors_info])

        return np.argmax(votes)

    def predict(self, X):
        predictions = []
        for x in X:
            pred = self._predict_one(x)
            predictions.append(pred)

        return np.array(predictions)

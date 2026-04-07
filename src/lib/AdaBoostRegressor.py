import numpy as np
import copy


class AdaBoostRegressor:
    def __init__(self, n_estimators, regressor, random_state=None, learning_rate=1):
        self.learning_rate = learning_rate
        self.regressor = regressor
        self.n_estimators = n_estimators
        self.predictors_info = None

    @staticmethod
    def weighted_error_rate(predicted_y, y, weights_examples):
        error = np.abs(y - predicted_y)
        error_max = np.max(error)

        if error_max == 0:
            error_max = 1e-10

        error = error / error_max
        r = np.sum(weights_examples * error)
        return r, error

    @staticmethod
    def predictor_weight(predicted_y, y, weights_examples, learning_rate):
        r, error = AdaBoostRegressor.weighted_error_rate(predicted_y, y, weights_examples)
        eps = 1e-10
        r = np.clip(r, eps, 1 - eps)
        beta = r / (1 - r)
        alpha = learning_rate * np.log(1 / beta)
        return error, alpha, beta, r

    @staticmethod
    def update_weights(weights_examples, beta, error):
        weights_examples *= beta ** (1 - error)
        weights_examples /= np.sum(weights_examples)
        return weights_examples

    @staticmethod
    def _train_regressor(X, y, regressor, weights_examples):
        tree_clf = copy.deepcopy(regressor)
        tree_clf.fit(X, y, sample_weights=weights_examples)
        return tree_clf

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        numExamples, numFeatures = X.shape

        weights_examples = np.ones(numExamples)
        weights_examples /= numExamples

        predictors = []

        for i in range(self.n_estimators):
            i_predictor = AdaBoostRegressor._train_regressor(X, y, self.regressor, weights_examples)
            predicted_y = i_predictor.predict(X)
            error, alpha, beta, r = AdaBoostRegressor.predictor_weight(predicted_y, y, weights_examples, self.learning_rate)
            if r >= 0.5:
                break
            predictors.append((i_predictor, alpha))
            weights_examples = AdaBoostRegressor.update_weights(weights_examples, beta, error)

        self.predictors_info = predictors

    def _predict_one(self, x):
        pred = []
        weights = []

        x = np.asarray(x)
        if x.ndim == 0:
            x = x.reshape(1, 1)
        elif x.ndim == 1:
            x = x.reshape(1, -1)

        for predictor, weight in self.predictors_info:
            pred.append(predictor.predict(x)[0])
            weights.append(weight)

        pred = np.array(pred)
        weights = np.array(weights)

        sorted_idx = np.argsort(pred)
        pred = pred[sorted_idx]
        weights = weights[sorted_idx]

        cum_weights = np.cumsum(weights) / np.sum(weights)

        return pred[np.searchsorted(cum_weights, 0.5)]

    def predict(self, X):
        predictions = []
        for x in X:
            pred = self._predict_one(x)
            predictions.append(pred)

        return np.array(predictions)

import numpy as np


class LogisticRegression:
    def __init__(self, ridge_alpha=0, lasso_alpha=0, l1_ratio=0, rate=0.0005):
        self.coef_ = None
        self.accuracy_ = None
        self.ridge_alpha = ridge_alpha
        self.lasso_alpha = lasso_alpha
        self.l1_ratio = l1_ratio
        self.rate = rate

    def softmax(t):
        sum_exp = np.sum(np.exp(t), axis=1, keepdims=True)
        prob_class = np.exp(t)/sum_exp
        return prob_class

    def gradient_descent(y_pred, y, numExamples, X, weights, rate, ridge_alpha, lasso_alpha, elastic_r):
        residual = y - y_pred  # numExamples, numClasses

        loss = -np.sum(y*np.log(y_pred))/numExamples + ((1 - elastic_r) * ridge_alpha * np.sum(weights**2))/numExamples + elastic_r * 2 * lasso_alpha * np.sum(abs(weights))

        sign_weights = np.sign(weights)
        sign_weights[weights == 0] = 0  # to fix non differentiable point

        grad_by_w = -(X.T)@(residual)/numExamples + (1-elastic_r)*(2*ridge_alpha*weights/numExamples) + elastic_r*(2*lasso_alpha*sign_weights)

        weights = weights - rate * grad_by_w

        return loss, weights

    def fit(self, X, y):
        numExamples, numWeights = X.shape
        X = np.c_[np.ones(numExamples), X]   # numExamples, numWeights
        numExamples, numWeights = X.shape
        k = np.unique(y).size         # numClasses
        weights = np.random.rand(numWeights, k)     # numWeights, numClasses
        y_pred = LogisticRegression.softmax(X @ weights)     # numExamples, numClasses
        # y_class = np.zeros((numExamples, k))     # numExamples, numClasses
        # for i in range(0, numExamples):
        #     y_class[i][y[i]-1] = 1
        y_class = np.eye(k)[y]

        i = 1
        prev_loss = float("inf")

        tolerance = 1e-8
        max_iters = 100000
        eps = 1e-15

        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss, weights = LogisticRegression.gradient_descent(y_pred, y_class, numExamples, X, weights, self.rate, self.ridge_alpha, self.lasso_alpha, self.l1_ratio)

        while abs(prev_loss - loss) > tolerance and i < max_iters:
            prev_loss = loss
            i += 1
            y_pred = LogisticRegression.softmax(X @ weights)
            y_pred = np.clip(y_pred, eps, 1 - eps)
            loss, weights = LogisticRegression.gradient_descent(y_pred, y_class, numExamples, X, weights, self.rate, self.ridge_alpha, self.lasso_alpha, self.l1_ratio)

        y_pred = LogisticRegression.softmax(X @ weights)
        y_pred_class = np.argmax(y_pred, axis=1)
        accuracy = np.mean(y_pred_class == y) * 100
        self.coef_ = weights
        self.accuracy_ = accuracy

    def predict(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        n, w = X.shape
        X = np.c_[np.ones(n), X]
        y_pred = X @ self.coef_
        y_pred_class = np.argmax(y_pred, axis=1)
        return y_pred_class

import numpy as np
import math


class LinearRegression:
    def __init__(self, ridge_alpha=0, lasso_alpha=0, l1_ratio=0, lnorm_type=2, rate=0.0005):
        self.coef_ = None
        self.intercept_ = None
        self.mse_ = None
        self.ridge_alpha = ridge_alpha
        self.lasso_alpha = lasso_alpha
        self.l1_ratio = l1_ratio
        self.lnorm_type = lnorm_type
        self.rate = rate

    def initialise_variables(numWeights):
        weights = np.random.rand(numWeights)
        bias = float(np.random.rand())
        return weights, bias

    def gradient_descent(grad_by_w, grad_by_b, weights, bias, rate):
        weights = weights - rate * grad_by_w
        bias = bias - rate * grad_by_b
        return weights, bias

    def loss_function(y_pred, y, l_type, numExamples, X, weights, ridge_alpha, lasso_alpha, elastic_r):
        residual = y - y_pred

        mse = math.sqrt(np.sum((abs(residual))**l_type)/numExamples)
        loss = np.sum((abs(residual))**l_type)/numExamples + ((1 - elastic_r) * ridge_alpha * np.sum(weights**2))/numExamples + elastic_r * 2 * lasso_alpha * np.sum(abs(weights))

        sign_grad = np.sign((residual))
        sign_weights = np.sign(weights)

        grad_by_w = -l_type*(X.T)@((abs(residual)**(l_type-1))*sign_grad)/numExamples + (1-elastic_r)*(2*ridge_alpha*weights/numExamples) + elastic_r*(2*lasso_alpha*sign_weights)
        grad_by_b = np.sum(-l_type*((abs(residual)**(l_type-1))*sign_grad))/numExamples

        return mse, loss, grad_by_w, grad_by_b

    def fit(self, X, y):
        numExamples, numWeights = X.shape
        weights, bias = LinearRegression.initialise_variables(numWeights)
        y_pred = X @ weights + bias

        i = 1
        tolerance = 1e-8
        max_iters = 100000
        prev_loss = float("inf")

        mse, loss, grad_by_w, grad_by_b = LinearRegression.loss_function(y_pred, y, self.lnorm_type, numExamples, X, weights, self.ridge_alpha, self.lasso_alpha, self.l1_ratio)

        while abs(prev_loss - loss) > tolerance and i < max_iters:
            prev_loss = loss
            weights, bias = LinearRegression.gradient_descent(grad_by_w, grad_by_b, weights, bias, self.rate)
            y_pred = X @ weights + bias

            # print(i, " mse:", mse)
            i += 1

            mse, loss, grad_by_w, grad_by_b = LinearRegression.loss_function(y_pred, y, self.lnorm_type, numExamples, X, weights, self.ridge_alpha, self.lasso_alpha, self.l1_ratio)
            # print(prev_loss, " ", loss)

        print("Converged at ", i, "th step")
        self.coef_ = weights
        self.intercept_ = bias
        self.mse_ = mse

    def predict(self, X_train):
        y_pred = X_train @ self.coef_ + self.intercept_
        return y_pred

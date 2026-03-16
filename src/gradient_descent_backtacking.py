import numpy as np
import math


class LinearRegression:
    def __init__(self, ridge_alpha=0, lasso_alpha=0, l1_ratio=0, lnorm_type=2, rate=0.0005):
        self.coef_ = None
        self.intercept_ = None
        self.rmse_ = None
        self.ridge_alpha = ridge_alpha
        self.lasso_alpha = lasso_alpha
        self.l1_ratio = l1_ratio
        self.lnorm_type = lnorm_type
        self.rate = rate

    def initialise_variables(numWeights):
        weights = np.random.rand(numWeights)
        bias = float(np.random.rand())
        return weights, bias

    def backtacking(self, new_weights, avg_weights, new_bias, avg_bias, loss, prev_loss, av_loss, av_grad_by_w, av_grad_by_b, grad_by_w, grad_by_b):
        if (prev_loss < loss):
            self.rate /= 10
            return avg_weights, avg_bias, av_grad_by_w, av_grad_by_b, av_loss
        if (loss < prev_loss) and (av_loss < loss):
            return avg_weights, avg_bias, av_grad_by_w, av_grad_by_b, av_loss
        return new_weights, new_bias, grad_by_w, grad_by_b, loss

    def gradient_descent(grad_by_w, grad_by_b, weights, bias, rate):
        weights = weights - rate * grad_by_w
        bias = bias - rate * grad_by_b
        return weights, bias

    def loss_function(y_pred, y, l_type, numExamples, X, weights, ridge_alpha, lasso_alpha, elastic_r):
        residual = y - y_pred

        rmse = math.sqrt(np.sum((abs(residual))**l_type)/numExamples)
        loss = np.sum((abs(residual))**l_type)/numExamples + ((1 - elastic_r) * ridge_alpha * np.sum(weights**2))/numExamples + elastic_r * 2 * lasso_alpha * np.sum(abs(weights))

        sign_grad = np.sign((residual))
        sign_weights = np.sign(weights)
        sign_weights[weights == 0] = 0  # to fix non differentiable point

        grad_by_w = -l_type*(X.T)@((abs(residual)**(l_type-1))*sign_grad)/numExamples + (1-elastic_r)*(2*ridge_alpha*weights/numExamples) + elastic_r*(2*lasso_alpha*sign_weights)
        grad_by_b = np.sum(-l_type*((abs(residual)**(l_type-1))*sign_grad))/numExamples

        return rmse, loss, grad_by_w, grad_by_b

    def fit(self, X, y):
        numExamples, numWeights = X.shape
        weights, bias = LinearRegression.initialise_variables(numWeights)
        y_pred = X @ weights + bias

        i = 1
        tolerance = 1e-8
        max_iters = 50000
        prev_loss = float("inf")

        rmse, loss, grad_by_w, grad_by_b = LinearRegression.loss_function(y_pred, y, self.lnorm_type, numExamples, X, weights, self.ridge_alpha, self.lasso_alpha, self.l1_ratio)

        while abs(prev_loss - loss) > tolerance and i < max_iters:
            prev_loss = loss
            new_weights, new_bias = LinearRegression.gradient_descent(grad_by_w, grad_by_b, weights, bias, self.rate)
            y_pred = X @ new_weights + new_bias

            i += 1

            rmse, loss, grad_by_w, grad_by_b = LinearRegression.loss_function(y_pred, y, self.lnorm_type, numExamples, X, new_weights, self.ridge_alpha, self.lasso_alpha, self.l1_ratio)

            # if loss < prev_loss:
            avg_weights = (new_weights+weights)/2
            avg_bias = (bias+new_bias)/2
            avg_y_pred = X @ avg_weights + avg_bias
            av_rmse, av_loss, av_grad_by_w, av_grad_by_b = LinearRegression.loss_function(avg_y_pred, y, self.lnorm_type, numExamples, X, avg_weights, self.ridge_alpha, self.lasso_alpha,
                                                                                          self.l1_ratio)

            weights, bias, grad_by_w, grad_by_b, loss = LinearRegression.backtacking(self, new_weights, avg_weights, new_bias, avg_bias, loss, prev_loss, av_loss, av_grad_by_w, av_grad_by_b,
                                                                                     grad_by_w, grad_by_b)

            if (i % 1000 == 0):
                print("At step ", i)

        self.coef_ = weights
        self.intercept_ = bias
        self.rmse_ = rmse

    def predict(self, X_train):
        y_pred = X_train @ self.coef_ + self.intercept_
        return y_pred

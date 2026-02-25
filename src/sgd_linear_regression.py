import numpy as np
import math


class SGDLinearRegression:
    def __init__(self, ridge_alpha=0, lasso_alpha=0, l1_ratio=0, lnorm_type=2, rate=0.001, max_iter=500, tol=1e-6, random_state=36):
        self.coef_ = None
        self.intercept_ = None
        self.rmse_ = None
        self.ridge_alpha = ridge_alpha
        self.lasso_alpha = lasso_alpha
        self.l1_ratio = l1_ratio
        self.lnorm_type = lnorm_type
        self.rate = rate
        self.tolerance = tol
        self.max_iters = max_iter
        self.random_state = random_state

    def learning_decay(decay):
        return 5/(decay + 50)

    def initialise_variables(numWeights):
        weights = np.random.rand(numWeights)
        bias = float(np.random.rand())
        return weights, bias

    def update_parameters(grad_by_w, grad_by_b, weights, bias, rate):
        weights = weights - rate * grad_by_w
        bias = bias - rate * grad_by_b
        return weights, bias

    def stochastic_gradient_descent(y_pred, y, l_type, numExamples, X, weights, ridge_alpha, lasso_alpha, elastic_r, bias, rate, epoch):
        for i in range(numExamples):
            random_idx = np.random.randint(0, numExamples)
            Xi = X[random_idx: random_idx+1]
            yi = y[random_idx: random_idx+1]
            residual_i = yi - (Xi @ weights + bias)
            sign_gradi = np.sign(residual_i)
            sign_weights = np.sign(weights)

            grad_by_w = -l_type*(Xi.T)@((abs(residual_i)**(l_type-1))*sign_gradi) + \
                (1-elastic_r)*(2*ridge_alpha*weights/numExamples) + elastic_r*(2*lasso_alpha*sign_weights)
            grad_by_b = np.sum(-l_type*((abs(residual_i)**(l_type-1))*sign_gradi))

            rate = SGDLinearRegression.learning_decay(epoch * numExamples + i)

            weights, bias = SGDLinearRegression.update_parameters(grad_by_w, grad_by_b, weights, bias, rate)

        loss = np.sum((abs(y - (X @ weights + bias)))**l_type)/numExamples + ((1 - elastic_r) * ridge_alpha * np.sum(weights**2))/numExamples +\
            elastic_r * 2 * lasso_alpha * np.sum(abs(weights))
        rmse = math.sqrt(np.sum((abs(y - (X @ weights + bias))**l_type)/numExamples))

        return rmse, loss, weights, bias, rate

    def fit(self, X, y):
        numExamples, numWeights = X.shape
        weights, bias = SGDLinearRegression.initialise_variables(numWeights)
        y_pred = X @ weights + bias

        i = 1
        prev_loss = float("inf")

        rmse, loss, weights, bias, self.rate = SGDLinearRegression.stochastic_gradient_descent(y_pred, y, self.lnorm_type, numExamples, X, weights, self.ridge_alpha, self.lasso_alpha,
                                                                                               self.l1_ratio, bias, self.rate, 1)

        while abs(prev_loss - loss) > self.tolerance and i < self.max_iters:
            prev_loss = loss
            y_pred = X @ weights + bias
            i += 1

            rmse, loss, weights, bias, self.rate = SGDLinearRegression.stochastic_gradient_descent(y_pred, y, self.lnorm_type, numExamples, X, weights,
                                                                                                   self.ridge_alpha, self.lasso_alpha, self.l1_ratio, bias, self.rate, i+1)

        print("Converged at ", i, "th step")
        self.coef_ = weights
        self.intercept_ = bias
        self.rmse_ = rmse

    def predict(self, X_train):
        y_pred = X_train @ self.coef_ + self.intercept_
        return y_pred

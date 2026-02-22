import numpy as np
import math


def initialise_variables(numWeights):
    weights = np.random.rand(numWeights)
    bias = float(np.random.rand())
    return weights, bias


def gradient_descent(grad_by_w, grad_by_b, weights, bias, rate):
    weights = weights - rate*grad_by_w
    bias = bias - rate*grad_by_b
    return weights, bias


def loss_function(y_pred, y, l_type, numExamples, X):
    residual = y - y_pred

    mse = math.sqrt(np.sum((abs(residual))**l_type)/numExamples)
    loss = np.sum((abs(residual))**l_type)

    sign_grad = np.sign((residual))

    grad_by_w = -l_type*(X.T)@((abs(residual)**(l_type-1))*sign_grad)/numExamples
    grad_by_b = np.sum(-l_type*((abs(residual)**(l_type-1))*sign_grad))/numExamples

    return mse, loss, grad_by_w, grad_by_b


def linear_regression(X, y, l_type=2, rate=0.0005):
    numExamples, numWeights = X.shape
    weights, bias = initialise_variables(numWeights)
    y_pred = X @ weights + bias

    i = 1
    tolerance = 0.001
    max_iters = 100000
    prev_loss = float("inf")

    mse, loss, grad_by_w, grad_by_b = loss_function(y_pred, y, l_type, numExamples, X)

    while abs(prev_loss - loss) > tolerance and i < max_iters:
        prev_loss = loss
        weights, bias = gradient_descent(grad_by_w, grad_by_b, weights, bias, rate)
        y_pred = X @ weights + bias

        # print(i, " mse:", mse)
        i += 1

        mse, loss, grad_by_w, grad_by_b = loss_function(y_pred, y, l_type, numExamples, X)
        # print(prev_loss, " ", loss)

    return weights, bias, mse

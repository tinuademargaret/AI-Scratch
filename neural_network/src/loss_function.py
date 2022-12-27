import numpy as np


def mean_squared_error(y_pred, y):
    loss = np.square(y_pred - y)
    return np.mean(loss)


def mean_squared_error_prime(y_pred, y):
    loss_derivative = 2 * (y_pred - y)
    return loss_derivative


def binary_cross_entropy(y_pred, y):
    log_loss = - (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return np.mean(log_loss)


def binary_cross_entropy_prime(y_pred, y):
    return - (y / y_pred + (1 - y) * 1 / (1 - y_pred))


def cross_entropy(y_pred, y):
    return -np.sum(y * np.log(y_pred))


def cross_entropy_prime(y_pred, y):
    return -np.sum(y * np.log(y_pred))

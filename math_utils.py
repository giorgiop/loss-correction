import numpy as np


def l2_norm(theta):
    return np.linalg.norm(theta)


def mean_op(X, y):
    return np.dot(y, X) / X.shape[0]

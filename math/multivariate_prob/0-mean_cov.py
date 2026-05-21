#!/usr/bin/env python3
"""Mean and Covariance calculation"""
import numpy as np


def mean_cov(X):
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")
    mean = X.sum(axis=0, keepdims=True) / n
    X_c = X - mean
    return mean, (X_c.T @ X_c) / (n - 1)

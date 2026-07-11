#!/usr/bin/env python3
"""Calculates the normalization constants of a matrix"""
import numpy as np


def normalization_constants(X):
    """
    Calculates the mean and standard deviation of each feature in a matrix.

    Parameters:
    X (numpy.ndarray): Matrix of shape (m, nx) to normalize
                       m is the number of data points
                       nx is the number of features

    Returns:
    tuple: (mean, std) of each feature, both of shape (nx,)
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return mean, std

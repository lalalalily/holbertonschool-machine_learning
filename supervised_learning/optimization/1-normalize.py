#!/usr/bin/env python3
"""Normalizes a matrix"""
import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix by subtracting the mean
    and dividing by the standard deviation.

    Parameters:
    X (numpy.ndarray): Matrix of shape (d, nx) to normalize
                       d is the number of data points
                       nx is the number of features
    m (numpy.ndarray): Matrix of shape (nx,) containing the mean of all
    s (numpy.ndarray): Matrix of shape (nx,) containing the std of all features

    Returns:
    numpy.ndarray: The normalized X matrix
    """
    return (X - m) / s

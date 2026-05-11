#!/usr/bin/env python3
"""Module to calculate likelihood of binomial data"""
import numpy as np


def likelihood(x, n, P):
    """Calculates the likelihood of obtaining data x and n for probabilities in P"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    # Using np.math.factorial for the scalar values
    n_fac = np.math.factorial(n)
    x_fac = np.math.factorial(x)
    nx_fac = np.math.factorial(n - x)
    combination = n_fac / (x_fac * nx_fac)
    # Vectorized likelihood calculation: C * p^x * (1-p)^(n-x)
    return combination * (P ** x) * ((1 - P) ** (n - x))

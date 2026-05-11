#!/usr/bin/env python3
"""Module to calculate intersection of binomial data and priors"""
import numpy as np


def intersection(x, n, P, Pr):
    """Calculates intersection """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an ")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")
    # Binomial Coefficient calculation
    fact = np.math.factorial
    n_cr = fact(n) / (fact(x) * fact(n - x))
    # Likelihood: P(Data | Hypothesis)
    l_hood = n_cr * (P ** x) * ((1 - P) ** (n - x))
    # Intersection: Likelihood * Prior
    return l_hood * Pr

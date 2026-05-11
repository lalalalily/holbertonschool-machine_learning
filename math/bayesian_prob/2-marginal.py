#!/usr/bin/env python3
import numpy as np


def marginal(x, n, P, Pr):
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        value = "x must be an integer that is greater than or equal to 0"
        raise ValueError(value)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    fact = np.math.factorial
    n_cr = fact(n) / (fact(x) * fact(n - x))
    l_hood = n_cr * (P ** x) * ((1 - P) ** (n - x))

    return np.sum(l_hood * Pr)

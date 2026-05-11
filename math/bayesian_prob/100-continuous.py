#!/usr/bin/env python3
"""Module to calculate continuous posterior probability"""
from scipy import special


def posterior(x, n, p1, p2):
    """
    Calculates the posterior probability that the probability of
    developing severe side effects falls within a range [p1, p2]
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        msg = "x must be an integer that is greater than or equal to 0"
        raise ValueError(msg)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(p1, float) or not (0 <= p1 <= 1):
        raise ValueError("p1 must be a float in the range [0, 1]")
    if not isinstance(p2, float) or not (0 <= p2 <= 1):
        raise ValueError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")
    # The posterior of a Binomial likelihood with a Uniform prior
    # follows a Beta distribution: Beta(alpha, beta)
    # alpha = x + 1, beta = n - x + 1
    a = x + 1
    b = n - x + 1
    # The probability in range [p1, p2] is the difference of the CDFs
    cdf_p2 = special.betainc(a, b, p2)
    cdf_p1 = special.betainc(a, b, p1)
    return cdf_p2 - cdf_p1

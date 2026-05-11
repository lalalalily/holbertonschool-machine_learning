#!/usr/bin/env python3
"""Module for Poisson distribution"""


class Poisson:
    """Represents a Poisson distribution"""
    def __init__(self, data=None, lambtha=1.):
        """Initializes the distribution"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Calculates PMF for k successes"""
        k = int(k)
        if k < 0:
            return 0
        e, lambtha, fact = 2.7182818285, self.lambtha, 1
        for i in range(1, k + 1):
            fact *= i
        return (e ** -lambtha * lambtha ** k) / fact

    def cdf(self, k):
        """Calculates CDF for k successes"""
        k = int(k)
        if k < 0:
            return 0
        cdf_value = 0
        for i in range(k + 1):
            cdf_value += self.pmf(i)
        return cdf_value

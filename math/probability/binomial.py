#!/usr/bin/env python3
"""Module for Binomial distribution"""


class Binomial:
    """Represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """Initializes the distribution"""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n, self.p = int(n), float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            var = sum((x - mean) ** 2 for x in data) / len(data)
            p_est = 1 - (var / mean)
            self.n = int(round(mean / p_est))
            self.p = float(mean / self.n)

    def pmf(self, k):
        """Calculates PMF for k successes"""
        k = int(k)
        if k < 0 or k > self.n:
            return 0

        def fact(n):
            """Helper for factorial"""
            f = 1
            for i in range(1, n + 1):
                f *= i
            return f
        n, p = self.n, self.p
        combination = fact(n) / (fact(k) * fact(n - k))
        return combination * (p ** k) * ((1 - p) ** (n - k))

    def cdf(self, k):
        """Calculates CDF for k successes"""
        k = int(k)
        if k < 0:
            return 0
        cdf_val = 0
        for i in range(k + 1):
            cdf_val += self.pmf(i)
        return cdf_val

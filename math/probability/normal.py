#!/usr/bin/env python3
"""Module for Normal distribution"""


class Normal:
    """Represents a normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initializes the distribution"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean, self.stddev = float(mean), float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            var = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = float(var ** 0.5)

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value"""
        pi, e = 3.1415926536, 2.7182818285
        exp = -0.5 * ((x - self.mean) / self.stddev) ** 2
        return (1 / (self.stddev * (2 * pi) ** 0.5)) * (e ** exp)

    def cdf(self, x):
        """Calculates the value of the CDF for a given x-value"""
        pi = 3.1415926536
        val = (x - self.mean) / (self.stddev * (2 ** 0.5))
        erf = (2 / (pi ** 0.5)) * (val - (val**3 / 3) + (val**5 / 10) -
                                   (val**7 / 42) + (val**9 / 216))
        return 0.5 * (1 + erf)

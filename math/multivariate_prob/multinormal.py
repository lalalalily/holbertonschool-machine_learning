#!/usr/bin/env python3
"""Multivariate Normal distribution"""
import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution."""

    def __init__(self, data):
        """Initializes the MultiNormal instance."""
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = data.mean(axis=1, keepdims=True)
        X_c = data - self.mean
        self.cov = (X_c @ X_c.T) / (n - 1)

    def pdf(self, x):
        """Calculates the PDF at a data point."""
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        d = self.mean.shape[0]
        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))
        diff = x - self.mean
        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        denom = np.sqrt(((2 * np.pi) ** d) * det)
        return float(np.exp(-0.5 * diff.T @ inv @ diff) / denom)

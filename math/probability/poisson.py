#!/usr/bin/env python3
"""Module containing the Poisson class to represent a Poisson distribution."""


class Poisson:
    """Represents a Poisson distribution."""

    def __init__(self, data=None, lambtha=1.):
        """Class constructor for Poisson distribution.        """
        if data is None:
            # Case 1: No data provided, use the given lambtha
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            # Case 2: Data provided, estimate lambtha from the list
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # Calculate mean of the data (lambtha)
            self.lambtha = float(sum(data) / len(data))

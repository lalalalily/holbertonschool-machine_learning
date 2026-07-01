#!/usr/bin/env python3
"""
Module containing the Neuron class
"""
import numpy as np


class Neuron:
    """
    Defines a single neuron performing binary classification
    """
    def __init__(self, nx):
        """
        Class constructor
        Args:
            nx: number of input features to the neuron
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Initialize weights with a random normal dist
        # in a 2D array (1, nx)
        self.W = np.random.normal(size=(1, nx))
        # Initialize bias to 0
        self.b = 0
        # Initialize activated output to 0
        self.A = 0

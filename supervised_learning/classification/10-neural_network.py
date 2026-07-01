#!/usr/bin/env python3
"""
Defines a neural network with one hidden layer performing binary classification
and includes cost calculation.
"""
import numpy as np


class NeuralNetwork:
    """
    Represents a neural network with one hidden layer
    """
    def __init__(self, nx, nodes):
        """
        Initializes the neural network
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Private hidden layer weights, biases, and activation output
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        # Private output neuron weights, bias, and activation output
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter for W1"""
        return self.__W1

    @property
    def b1(self):
        """Getter for b1"""
        return self.__b1

    @property
    def A1(self):
        """Getter for A1"""
        return self.__A1

    @property
    def W2(self):
        """Getter for W2"""
        return self.__W2

    @property
    def b2(self):
        """Getter for b2"""
        return self.__b2

    @property
    def A2(self):
        """Getter for A2"""
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        Parameters:
        Y: numpy.ndarray with shape (1, m) containing the correct labels
        A: numpy.ndarray with shape (1, m) containing the activated output

        Returns:
        The cost value
        """
        m = Y.shape[1]

        # Logistic regression loss / Cross-entropy cost calculation
        # Uses 1.0000001 - A to safeguard against log(0)
        loss = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = -1 / m * np.sum(loss)

        return cost

#!/usr/bin/env python3
"""
Module containing the upgraded Neuron class with diagnostic graphing.
"""
import matplotlib.pyplot as plt
import numpy as np


class Neuron:
    """
    Defines a single neuron performing binary classification.
    """

    def __init__(self, nx):
        """
        Initializes the neuron.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Weights vector getter."""
        return self.__W

    @property
    def b(self):
        """Bias getter."""
        return self.__b

    @property
    def A(self):
        """Activated output getter."""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates forward propagation.
        """
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates logistic regression cost.
        """
        m = Y.shape[1]
        loss = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = -1 / m * np.sum(loss)
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates predictions.
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one iteration of gradient descent.
        """
        m = Y.shape[1]
        dZ = A - Y
        dW = (1 / m) * np.dot(dZ, X.T)
        db = (1 / m) * np.sum(dZ)
        self.__W = self.__W - (alpha * dW)
        self.__b = self.__b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the neuron while supporting verbose feedback and line plots.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        steps_recorded = []
        costs_recorded = []

        i = 0
        while i <= iterations:
            # Evaluate model state prior to standard training iteration step
            if i < iterations:
                A = self.forward_prop(X)
                c = self.cost(Y, A)
            else:
                # Final pass evaluation data capture
                A, c = self.evaluate(X, Y)

            # Record metrics at index 0, at step milestones, and at the end
            if i == 0 or i == iterations or i % step == 0:
                if verbose:
                    print("Cost after {} iterations: {}".format(i, c))
                if graph:
                    steps_recorded.append(i)
                    costs_recorded.append(c)

            if i < iterations:
                self.gradient_descent(X, Y, A, alpha)

            i += 1

        if graph:
            plt.plot(steps_recorded, costs_recorded, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

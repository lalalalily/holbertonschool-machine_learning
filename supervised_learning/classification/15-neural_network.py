#!/usr/bin/env python3
"""
Defines a neural network with one hidden layer performing binary classification
and includes upgraded iterative training functionality with logging and graphs.
"""
import matplotlib.pyplot as plt
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

        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter W1"""
        return self.__W1

    @property
    def b1(self):
        """Getter b1"""
        return self.__b1

    @property
    def A1(self):
        """Getter A1"""
        return self.__A1

    @property
    def W2(self):
        """Getter W2"""
        return self.__W2

    @property
    def b2(self):
        """Getter b2"""
        return self.__b2

    @property
    def A2(self):
        """Getter A2"""
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the network
        using the sigmoid activation function.
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.
        Uses 1.0000001 - A instead of 1 - A to avoid division by zero.
        """
        m = Y.shape[1]
        loss = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = -1 / m * np.sum(loss)
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates predictions made by the network.
        """
        _, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        prediction = np.where(A2 >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network.
        """
        m = Y.shape[1]

        dZ2 = A2 - Y
        dW2 = (1 / m) * np.matmul(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.matmul(self.__W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = (1 / m) * np.matmul(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the neural network over a set amount of cycles.

        Parameters:
        X: numpy.ndarray input features
        Y: numpy.ndarray target labels
        iterations: total training loops to process
        alpha: the step size modifier (learning rate)
        verbose: bool deciding whether to print progress metrics
        graph: bool deciding whether to render a matplotlib visual plot
        step: print/record metrics gap step interval
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

        for i in range(iterations):
            # Calculate current performance state before updating
            if verbose or graph:
                if i % step == 0:
                    _, cur_A2 = self.forward_prop(X)
                    cur_cost = self.cost(Y, cur_A2)
                    if verbose:
                        print("Cost after {} iterations: {}".format(
                            i, cur_cost))
                    if graph:
                        steps_recorded.append(i)
                        costs_recorded.append(cur_cost)

            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)

        # Include final iteration step information
        if verbose or graph:
            _, final_A2 = self.forward_prop(X)
            final_cost = self.cost(Y, final_A2)
            if verbose:
                print("Cost after {} iterations: {}".format(
                    iterations, final_cost))
            if graph:
                steps_recorded.append(iterations)
                costs_recorded.append(final_cost)

            if graph:
                plt.plot(steps_recorded, costs_recorded, 'b-')
                plt.xlabel('iteration')
                plt.ylabel('cost')
                plt.title('Training Cost')
                plt.show()

        return self.evaluate(X, Y)

def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one iteration of gradient descent on the neuron.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).
            Y (numpy.ndarray): Correct labels with shape (1, m).
            A (numpy.ndarray): Activated output of the neuron with shape (1, m).
            alpha (float): The learning rate.
        """
        m = Y.shape[1]
        # Calculate the error in prediction (dZ)
        dZ = A - Y
        # Calculate gradients for weights (dW) and bias (db)
        dW = (1 / m) * np.dot(dZ, X.T)
        db = (1 / m) * np.sum(dZ)
        # Update the private weight vector and bias using gradient descent
        self.__W = self.__W - (alpha * dW)
        self.__b = self.__b - alpha * db

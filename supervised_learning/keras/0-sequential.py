#!/usr/bin/env python3
"""
Defines a function that builds a neural network with the Keras library
Using the Sequential API without the Input class.
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library.

    Args:
        nx: number of input features to the network
        layers: list containing the number of nodes in each layer
        activations: list containing the activation functions for each layer
        lambtha: L2 regularization parameter
        keep_prob: probability that a node will be kept for dropout

    Returns:
        The Keras model
    """
    model = K.Sequential()

    # Configure L2 regularization using the K alias
    regularizer = K.regularizers.l2(lambtha)

    for i in range(len(layers)):
        if i == 0:
            # First layer specifies the input shape directly
            model.add(K.layers.Dense(
                units=layers[i],
                activation=activations[i],
                kernel_regularizer=regularizer,
                input_shape=(nx,)
            ))
        else:
            model.add(K.layers.Dense(
                units=layers[i],
                activation=activations[i],
                kernel_regularizer=regularizer
            ))

        # Add Dropout layer after each hidden layer
        if i < len(layers) - 1:
            # Keras Dropout takes the drop rate (1 - keep_prob)
            model.add(K.layers.Dropout(rate=1 - keep_prob))

    return model

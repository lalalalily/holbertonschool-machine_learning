#!/usr/bin/env python3
"""
Defines a function that builds a neural network with the Keras library
Using the Functional API instead of the Sequential API.
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library using the Functional API.

    Args:
        nx: number of input features to the network
        layers: list containing the number of nodes in each layer
        activations: list containing the activation functions for each layer
        lambtha: L2 regularization parameter
        keep_prob: probability that a node will be kept for dropout

    Returns:
        The Keras model
    """
    # Define the input layer explicitly for the Functional API
    inputs = K.Input(shape=(nx,))

    # Configure L2 regularization
    regularizer = K.regularizers.l2(lambtha)

    # Connect the first hidden layer to the inputs
    X = K.layers.Dense(
        units=layers[0],
        activation=activations[0],
        kernel_regularizer=regularizer
    )(inputs)

    # Add dropout to the first layer if there are more layers to come
    if len(layers) > 1:
        X = K.layers.Dropout(rate=1 - keep_prob)(X)

    # Loop through the remaining layers
    for i in range(1, len(layers)):
        X = K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=regularizer
        )(X)

        # Apply dropout to all hidden layers except the final output layer
        if i < len(layers) - 1:
            X = K.layers.Dropout(rate=1 - keep_prob)(X)

    # Create the model by defining the inputs and final outputs
    model = K.Model(inputs=inputs, outputs=X)

    return model

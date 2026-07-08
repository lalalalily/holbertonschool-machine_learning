#!/usr/bin/env python3
"""
Defines a function that sets up Adam optimization for a Keras model
with categorical crossentropy loss and accuracy metrics.
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Sets up Adam optimization for a Keras model.

    Args:
        network: the Keras model to optimize
        alpha: the learning rate
        beta1: the first Adam optimization parameter
        beta2: the second Adam optimization parameter

    Returns:
        None
    """
    # Instantiate the Adam optimizer with the given parameters
    optimizer = K.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2
    )

    # Compile the network with the optimizer, loss function, and metrics
    network.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

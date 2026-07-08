#!/usr/bin/env python3
"""
Defines a function that trains a model using mini-batch gradient descent
and validates it on validation data.
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    Trains a Keras model using mini-batch gradient descent.

    Args:
        network: the Keras model to train
        data: numpy.ndarray of shape (m, nx) containing the input data
        labels: one-hot numpy.ndarray of shape (m, classes) with the labels
        batch_size: size of the batch used for mini-batch gradient descent
        epochs: number of passes through the data
        validation_data: data to validate the model with, as a tuple of
                         (X_valid, Y_valid), or None
        verbose: boolean that determines if output should be printed
        shuffle: boolean that determines whether to shuffle batches each epoch

    Returns:
        The History object generated after training the model.
    """
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        verbose=verbose,
        shuffle=shuffle
    )

    return history

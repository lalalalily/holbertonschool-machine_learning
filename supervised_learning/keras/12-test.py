#!/usr/bin/env python3
"""
Module to evaluate a neural network's performance on test data
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network model and evaluates its performance.

    Parameters:
    network: The network model to test.
    data: The input data to test the model with.
    labels: The correct one-hot labels of the data.
    verbose: A boolean determining if output should be printed
             during the testing process.

    Returns:
    A list containing the loss and accuracy of the model, respectively.
    """
    return network.evaluate(data, labels, verbose=verbose)

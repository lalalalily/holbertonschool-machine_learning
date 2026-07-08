#!/usr/bin/env python3
"""
Module to generate predictions using a neural network
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network model.

    Parameters:
    network: The network model to make the prediction with.
    data: The input data to make the prediction with.
    verbose: A boolean determining if output should be printed
             during the prediction process.

    Returns:
    The prediction array for the input data.
    """
    return network.predict(data, verbose=verbose)

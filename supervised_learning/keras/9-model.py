#!/usr/bin/env python3
"""
Defines functions to save and load an entire Keras model.
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire Keras model to a specified file path.

    Args:
        network: the Keras model to save
        filename: the path of the file that the model should be saved to

    Returns:
        None
    """
    K.models.save_model(network, filename)


def load_model(filename):
    """
    Loads an entire Keras model from a specified file path.

    Args:
        filename: the path of the file that the model should be loaded from

    Returns:
        The loaded Keras model
    """
    return K.models.load_model(filename)

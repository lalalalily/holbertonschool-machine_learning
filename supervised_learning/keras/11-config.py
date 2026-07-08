#!/usr/bin/env python3
"""
Module to save and load model configurations
"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model's configuration in JSON format.

    Parameters:
    network: The model whose configuration should be saved.
    filename: The path of the file that the configuration should be saved to.

    Returns:
    None
    """
    model_json = network.to_json()
    with open(filename, "w", encoding="utf-8") as json_file:
        json_file.write(model_json)


def load_config(filename):
    """
    Loads a model with a specific configuration from a JSON file.

    Parameters:
    filename: The path of the file containing the model's configuration.

    Returns:
    The instantiated model with initialized weights.
    """
    with open(filename, "r", encoding="utf-8") as json_file:
        model_json = json_file.read()

    return K.models.model_from_json(model_json)

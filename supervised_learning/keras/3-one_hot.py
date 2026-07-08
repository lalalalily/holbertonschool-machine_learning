#!/usr/bin/env python3
"""
Defines a function that converts a label vector into a one-hot matrix.
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.

    Args:
        labels: label vector to convert
        classes: total number of classes. If None, it will be inferred
                 automatically as the max value in labels + 1

    Returns:
        The one-hot matrix
    """
    return K.utils.to_categorical(labels, num_classes=classes)

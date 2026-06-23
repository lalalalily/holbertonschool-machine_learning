#!/usr/bin/env python3
"""
Defines a function that calculates precision for each class
"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.
    """
    true_positives = np.diag(confusion)
    predicted_positives = np.sum(confusion, axis=0)
    return true_positives / predicted_positives

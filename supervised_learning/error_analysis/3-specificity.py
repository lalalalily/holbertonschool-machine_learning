#!/usr/bin/env python3
"""
Defines a function that calculates specificity for each class
"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix.
    """
    total = np.sum(confusion)
    tp = np.diag(confusion)
    actual_positives = np.sum(confusion, axis=1)
    predicted_positives = np.sum(confusion, axis=0)
    fp = predicted_positives - tp
    fn = actual_positives - tp
    tn = total - (tp + fp + fn)
    actual_negatives = tn + fp
    return tn / actual_negatives

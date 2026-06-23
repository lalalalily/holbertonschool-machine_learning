#!/usr/bin/env python3
"""
Module containing the function create_confusion_matrix
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix from one-hot encoded labels/predictions.

    Args:
        labels (numpy.ndarray): One-hot encoded matrix of shape (m, classes)
                                containing the correct labels.
        logits (numpy.ndarray): One-hot encoded matrix of shape (m, classes)
                                containing the predicted labels.

    Returns:
        numpy.ndarray: Confusion matrix of shape (classes, classes) where
                       rows represent correct labels and columns represent
                       predictions.
    """
    return np.dot(labels.T, logits)

#!/usr/bin/env python3
"""Simple policy function using softmax."""
import numpy as np


def policy(matrix, weight):
    """
    Computes the policy with a weight of a matrix.

    Args:
        matrix: state matrix
        weight: weight matrix

    Returns:
        the policy (probabilities of each action) as a numpy array
    """
    z = matrix.dot(weight)
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

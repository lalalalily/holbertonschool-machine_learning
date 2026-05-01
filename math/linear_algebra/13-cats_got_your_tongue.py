#!/usr/bin/env python3
"""Module for concatenating numpy arrays."""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatenate two numpy arrays along a specific axis.
    Args:
        mat1: A numpy.ndarray
        mat2: A numpy.ndarray
        axis: 0 to concatenate vertically (rows)
    Returns:
        A new concatenated numpy.ndarray
    """
    return np.concatenate((mat1, mat2), axis=axis)

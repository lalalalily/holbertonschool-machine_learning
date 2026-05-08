#!/usr/bin/env python3
"""Module to calculate the definiteness of a matrix"""
import numpy as np


def definiteness(matrix):
    """Calculates the definiteness of a matrix"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    if not np.allclose(matrix, matrix.T):
        return None
    try:
        val = np.linalg.eigvals(matrix)
        pos = np.all(val > 1e-10)
        neg = np.all(val < -1e-10)
        semi_pos = np.all(val >= -1e-10)
        semi_neg = np.all(val <= 1e-10)
        if pos:
            return "Positive definite"
        if neg:
            return "Negative definite"
        if semi_pos:
            return "Positive semi-definite"
        if semi_neg:
            return "Negative semi-definite"
        return "Indefinite"
    except Exception:
        return None

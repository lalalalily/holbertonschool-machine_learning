#!/usr/bin/env python3
"""Normalizes an unactivated output using Batch Normalization"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network using
    batch normalization.

    Parameters:
    Z (numpy.ndarray): Matrix of shape (m, n) to be normalized
    gamma (numpy.ndarray): Matrix of shape (1, n) containing scales
    beta (numpy.ndarray): Matrix of shape (1, n) containing offsets
    epsilon (float): Small number to avoid division by zero

    Returns:
    numpy.ndarray: The normalized Z matrix
    """
    # Compute mean and variance along data points (axis 0)
    mean = np.mean(Z, axis=0
    variance = np.var(Z, axis=0)

    # Standardize the unactivated matrix
    Z_hat = (Z - mean) / np.sqrt(variance + epsilon)

    # Scale and shift using the learnable parameters
    Z_norm = gamma * Z_hat + beta

    return Z_norm

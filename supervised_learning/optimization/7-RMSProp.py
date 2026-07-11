#!/usr/bin/env python3
"""Updates a variable using the RMSProp optimization algorithm"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm.

    Parameters:
    alpha (float): The learning rate
    beta2 (float): The RMSProp weight (discounting factor)
    epsilon (float): A small number to avoid division by zero
    var (numpy.ndarray): The variable to be updated
    grad (numpy.ndarray): The gradient of var
    s (numpy.ndarray): The previous second moment of var

    Returns:
    tuple: (updated_var, new_s)
    """
    # Calculate the new second raw moment vector (exponential moving average)
    s_new = beta2 * s + (1 - beta2) * (grad ** 2)

    # Update the variable weights
    var_new = var - (alpha / (np.sqrt(s_new) + epsilon)) * grad

    return var_new, s_new

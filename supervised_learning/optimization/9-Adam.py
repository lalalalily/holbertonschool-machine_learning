#!/usr/bin/env python3
"""Updates a variable using the Adam optimization algorithm"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable using the Adam optimization algorithm.

    Parameters:
    alpha (float): The learning rate
    beta1 (float): The weight used for the first moment
    beta2 (float): The weight used for the second moment
    epsilon (float): A small number to avoid division by zero
    var (numpy.ndarray): The variable to be updated
    grad (numpy.ndarray): The gradient of var
    v (numpy.ndarray): The previous first moment of var
    s (numpy.ndarray): The previous second moment of var
    t (int): The time step used for bias correction

    Returns:
    tuple: (updated_var, new_v, new_s)
    """
    # 1. Update biased first moment estimate
    v_new = beta1 * v + (1 - beta1) * grad

    # 2. Update biased second raw moment estimate
    s_new = beta2 * s + (1 - beta2) * (grad ** 2)

    # 3. Compute bias-corrected first moment estimate
    v_corrected = v_new / (1 - (beta1 ** t))

    # 4. Compute bias-corrected second raw moment estimate
    s_corrected = s_new / (1 - (beta2 ** t))

    # 5. Update parameters in-place
    var -= (alpha / (np.sqrt(s_corrected) + epsilon)) * v_corrected

    return var, v_new, s_new

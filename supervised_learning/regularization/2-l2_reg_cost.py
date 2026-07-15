#!/usr/bin/env python3
"""
Module containing the l2_reg_cost function for Keras models.
"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization.
    Parameters:
    cost: A tensor containing the cost of the network without L2 regularization.
    model: A Keras model that includes layers with L2 regularization.
    Returns:
    A tensor containing the total cost accounting for L2 regularization.
    """
    total_cost = cost + tf.add_n(model.losses)
    return total_cost

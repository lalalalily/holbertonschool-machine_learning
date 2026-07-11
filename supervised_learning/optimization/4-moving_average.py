#!/usr/bin/env python3
"""Calculates the weighted moving average with bias correction"""


def moving_average(data, beta):
    """
    Calculates the exponentially weighted moving average of a data set
    using bias correction.

    Parameters:
    data (list): List of data points
    beta (float): The weight used for the moving average

    Returns:
    list: A list containing the moving averages of the data
    """
    moving_averages = []
    v = 0

    for t, theta in enumerate(data, start=1):
        # Update the exponentially weighted moving average
        v = beta * v + (1 - beta) * theta

        # Apply bias correction
        v_corrected = v / (1 - (beta ** t))
        moving_averages.append(v_corrected)

    return moving_averages

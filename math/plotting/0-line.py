#!/usr/bin/env python3
"""
This module contains a function to plot a mathematical cubic line graph.
"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Plots y = x^3 as a solid red line with an x-axis ranging from 0 to 10.
    """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    # Generate x values corresponding to the indices of y (0 through 10)
    x = np.arange(0, 11)

    # Plot y as a solid red line ('r-')
    plt.plot(x, y, 'r-')

    # Explicitly set the x-axis limits
    plt.xlim(0, 10)

    # Display the plot
    plt.show()

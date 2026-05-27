#!/usr/bin/env python3
"""
This module contains a function to plot a 2D scatter plot representing
mountain elevation with a gradient color map.
"""
import numpy as np
import matplotlib.pyplot as plt


def gradient():
    """
    Creates a scatter plot of sampled mountain elevations, mapping the
    height (z) to a colorbar gradient.
    """
    np.random.seed(5)

    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))
    plt.figure(figsize=(6.4, 4.8))

    # Create the scatter plot with z mapping to color
    scatter = plt.scatter(x, y, c=z, cmap='viridis')

    # Add labels and title
    plt.xlabel('x coordinate (m)')
    plt.ylabel('y coordinate (m)')
    plt.title('Mountain Elevation')

    # Create and label the colorbar
    colorbar = plt.colorbar(scatter)
    colorbar.set_label('elevation (m)')

    plt.show()

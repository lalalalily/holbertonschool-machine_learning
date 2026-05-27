#!/usr/bin/env python3
"""
Defines a scatter plot visualization function for height and weight data
"""
import numpy as np
import matplotlib.pyplot as plt


def scatter():
    """
    Plots a scatter plot of Men's Height vs Weight using magenta points
    and appropriate axis labeling.
    """
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180
    plt.figure(figsize=(6.4, 4.8))

    # Plot the data points using magenta color
    plt.scatter(x, y, color='m')

    # Configure the titles and labels
    plt.xlabel('Height (in)')
    plt.ylabel('Weight (lbs)')
    plt.title("Men's Height vs Weight")

    # Render the visualization
    plt.show()

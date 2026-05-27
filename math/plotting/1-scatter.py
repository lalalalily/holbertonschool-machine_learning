#!/usr/bin/env python3
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

    # Plot the scatter data with magenta points ('m')
    plt.scatter(x, y, color='m')

    # Add descriptive axis labels and chart title
    plt.xlabel('Height (in)')
    plt.ylabel('Weight (lbs)')
    plt.title("Men's Height vs Weight")

    # Display the final visualization layout
    plt.show()

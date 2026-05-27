#!/usr/bin/env python3
"""
Defines a function that plots two exponential decay curves on the same graph
"""
import numpy as np
import matplotlib.pyplot as plt


def two():
    """
    Plots the exponential decay of C-14 and Ra-226 with specific colors,
    line styles, axis limits, titles, and a legend.
    """
    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730
    t2 = 1600
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)
    plt.figure(figsize=(6.4, 4.8))

    # Plot y1 as a dashed red line with label C-14
    plt.plot(x, y1, 'r--', label='C-14')

    # Plot y2 as a solid green line with label Ra-226
    plt.plot(x, y2, 'g-', label='Ra-226')

    # Configure the axis labels and title
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.title('Exponential Decay of Radioactive Elements')

    # Force the display boundaries for the viewports
    plt.xlim(0, 20000)
    plt.ylim(0, 1)

    # Place the legend explicitly in the upper right-hand corner
    plt.legend(loc='upper right')

    # Render the visualization
    plt.show()

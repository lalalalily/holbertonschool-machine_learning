#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Plots a line graph with a solid red line and x-axis
    """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    # Generate x values corresponding to y indices (0 through 10)
    x = np.arange(0, 11)

    # Plot y as a solid red line ('r-')
    plt.plot(x, y, 'r-')

    # Force the x-axis limits to range exactly from 0 to 10
    plt.xlim(0, 10)

    # Display the final graph layout
    plt.show()

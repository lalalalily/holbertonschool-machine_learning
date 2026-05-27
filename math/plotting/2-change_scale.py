#!/usr/bin/env python3
"""
Defines a function that plots exponential decay with a logarithmic scale
"""
import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """
    Plots the exponential decay of C-14 using a logarithmic scale on the y-axis
    with custom axis labels, title, and x-axis limits.
    """
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)
    plt.figure(figsize=(6.4, 4.8))

    # Plot the line graph
    plt.plot(x, y)

    # Set the y-axis to a logarithmic scale
    plt.yscale('log')

    # Force the x-axis limits to range exactly from 0 to 28650
    plt.xlim(0, 28650)

    # Add labels and title
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.title('Exponential Decay of C-14')

    # Display the plot
    plt.show()

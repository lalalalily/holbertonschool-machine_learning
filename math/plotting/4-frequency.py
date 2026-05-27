#!/usr/bin/env python3
"""Module that contains a function to plot student grade frequency."""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """Plots a histogram of student scores for a project."""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # Bins every 10 units. Since grades naturally fall between 0 and 100, 
    # we define the bins range to cover the data.
    bins = np.arange(0, 101, 10)

    # Plot histogram with black outlines
    plt.hist(student_grades, bins=bins, edgecolor='black')

    # Label the axes and title exactly as requested
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    
    # Explicitly bound X-axis to the exact bin limits to avoid clipping/extra whitespace
    plt.xlim(0, 100)

    plt.show()

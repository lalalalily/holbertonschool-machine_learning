#!/usr/bin/env python3
"""Module that contains a function to plot student grade frequency."""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """Plots a histogram of student scores for a project."""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # Define the bins every 10 units (from 0 to 100 based on typical grades)
    # np.arange(0, 101, 10) creates: [0, 10, 20, ..., 100]
    bins = np.arange(0, 101, 10)

    # Plot the histogram with black outlines around the bars
    plt.hist(student_grades, bins=bins, edgecolor='black')

    # Set X and Y axis labels
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')

    # Explicitly set the X-axis ticks to match the 10-unit bins
    plt.xticks(bins)
  
    # Ensure the X-axis view limits tightly wrap around the bin range
    plt.xlim(0, 100)

    # Set the title
    plt.title('Project A')

    plt.show()

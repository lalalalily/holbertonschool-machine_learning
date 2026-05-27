#!/usr/bin/env python3
"""
This module contains a function to visualize student grades using a histogram.
"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Plots a histogram representing the frequency of student grades
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # Define bins every 10 units from 0 to 100
    bins = np.arange(0, 110, 10)

    # Plot the histogram with black outlines
    plt.hist(student_grades, bins=bins, edgecolor='black')

    # Add labels and title
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')

    # Explicitly set the axis intervals and limits to match the reference
    plt.xticks(bins)
    plt.xlim(0, 100)
    plt.ylim(0, 30)

    plt.show()

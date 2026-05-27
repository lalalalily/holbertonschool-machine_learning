#!/usr/bin/env python3
"""
Defines a function that plots a histogram of student grades
"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Plots a histogram of student scores for Project A with bins every 10 units,
    black outlines on the bars, and customized axis labels and titles.
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # Define bins every 10 units from 0 to 100 to align perfectly with the axis
    bins = np.arange(0, 110, 10)

    # Plot the histogram with black outlines around the bars using edgecolor
    plt.hist(student_grades, bins=bins, edgecolor='black')

    # Add descriptive labels and title
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')

    # Restrict x-axis limits to wrap the bins cleanly
    plt.xlim(0, 100)

    # Force x-axis ticks to display every 10 units exactly at the bin boundaries
    plt.xticks(bins)

    # Display the final plot layout
    plt.show()

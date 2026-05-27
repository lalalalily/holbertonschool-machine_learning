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

    # Explicitly define bins from 0 to 100 to keep the math uniform
    bins = list(range(0, 110, 10))

    # Plot the histogram with black outlines around the bars
    plt.hist(student_grades, bins=bins, edgecolor='black')

    # Add descriptive labels and title
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')

    # Force x-axis limits to match the reference graph's framing (40 to 100)
    plt.xlim(40, 100)

    # Set explicit ticks at every 10 units from 40 to 100
    plt.xticks(list(range(40, 110, 10)))

    # Display the final plot layout
    plt.show()

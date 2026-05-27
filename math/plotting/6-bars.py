#!/usr/bin/env python3
"""
Defines a function that plots a stacked bar chart of fruit distribution
"""
import numpy as np
import matplotlib.pyplot as plt


def fruits():
    """
    Plots a stacked bar chart of fruit consumption per person
    with custom colors, widths, legends, and specific axis styling.
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    
    # Assign fruit categories from the matrix rows
    apples = fruit[0]
    bananas = fruit[1]
    oranges = fruit[2]
    peaches = fruit[3]
    
    persons = ['Farrah', 'Fred', 'Felicia']
    x = np.arange(len(persons))
    width = 0.5

    plt.figure(figsize=(6.4, 4.8))

    # Plot each fruit layer, stacking them on top of previous layers using 'bottom'
    plt.bar(x, apples, width=width, color='red', label='apples')
    plt.bar(x, bananas, width=width, bottom=apples,
            color='yellow', label='bananas')
    plt.bar(x, oranges, width=width, bottom=apples + bananas,
            color='#ff8000', label='oranges')
    plt.bar(x, peaches, width=width, bottom=apples + bananas + oranges,
            color='#ffe5b4', label='peaches')

    # Apply axis text and title formatting
    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    
    # Configure horizontal axes ticks and names
    plt.xticks(x, persons)
    
    # Force vertical axis boundaries between 0 and 80 with intervals of 10
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))

    # Add the legend tracking categories
    plt.legend()

    # Render the final visualization layout
    plt.show()

#!/usr/bin/env python3
"""
Defines a function that plots a stacked bar chart of fruit distribution
"""
import numpy as np
import matplotlib.pyplot as plt


def fruits():
    """
    Plots a stacked bar chart of fruit consumption per person
    with exact color profiles, widths, ticks, and layout metrics.
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    
    # Split individual data vectors out of the random seed array matrix
    apples = fruit[0]
    bananas = fruit[1]
    oranges = fruit[2]
    peaches = fruit[3]
    
    persons = ['Farrah', 'Fred', 'Felicia']
    x = np.arange(len(persons))
    width = 0.5

    plt.figure(figsize=(6.4, 4.8))

    # Construct the sequential stacked bar graph bars
    plt.bar(x, apples, width=width, color='red', label='apples')
    plt.bar(x, bananas, width=width, bottom=apples,
            color='yellow', label='bananas')
    plt.bar(x, oranges, width=width, bottom=apples + bananas,
            color='#ff8000', label='oranges')
    plt.bar(x, peaches, width=width, bottom=apples + bananas + oranges,
            color='#ffe5b4', label='peaches')

    # Apply specific labeling and title keys matching check requirements
    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    
    # Setup axis spacing indices and text ticks
    plt.xticks(x, persons)
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))

    # Add descriptive mapping label legends
    plt.legend()

    # Output plot frame window
    plt.show()

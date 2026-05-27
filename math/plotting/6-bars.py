#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def bars():
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4,3))
    plt.figure(figsize=(6.4, 4.8))

    # --- Your Code Here ---
    
    # Define names for X-axis ticks
    people = ['Farrah', 'Fred', 'Felicia']
    
    # Define fruit types, their hex colors, and tracking for the stack bottom
    fruit_types = ['apples', 'bananas', 'oranges', 'peaches']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
    
    # Initialize the bottom of the stack at zero for each person
    bottoms = np.zeros(3)
    
    # Plot each row (fruit) stacked on top of the previous ones
    for i in range(len(fruit)):
        plt.bar(people, fruit[i], width=0.5, bottom=bottoms, color=colors[i], label=fruit_types[i])
        bottoms += fruit[i] # Update the bottom position for the next layer
        
    # Set the title and labels
    plt.title('Number of Fruit per Person')
    plt.ylabel('Quantity of Fruit')
    
    # Set Y-axis range and step ticks
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    
    # Display the legend
    plt.legend()
    
    # --- End of Code ---

    plt.show()

#!/usr/bin/env python3
"""Module for calculating matrix shape."""


def matrix_shape(matrix):
    """
    Calculate the shape of a matrix.
    
    Args:
        matrix: A list (can be nested at any depth)
    
    Returns:
        A list of integers representing the shape
    """
    shape = []
    current = matrix
    
    while isinstance(current, list):
        shape.append(len(current))
        if len(current) == 0:
            break
        current = current[0]
    
    return shape
#!/usr/bin/env python3
"""Module for transposing matrices."""


def matrix_transpose(matrix):
    """
    Return the transpose of a 2D matrix.
    Args:
        matrix: A 2D list (rows x columns)
    Returns:
        A new transposed matrix (columns x rows)
    """
    rows = len(matrix)
    cols = len(matrix[0])
    transposed = []
    for col in range(cols):
        new_row = []
        for row in range(rows):
            new_row.append(matrix[row][col])
        transposed.append(new_row)
    return transposed

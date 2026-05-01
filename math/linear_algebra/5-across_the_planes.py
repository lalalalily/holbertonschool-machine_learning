#!/usr/bin/env python3
"""Module for adding 2D matrices element-wise."""


def add_matrices2D(mat1, mat2):
    """Add two 2D matrices element-wise.
    Args:
        mat1: A 2D list of ints/floats
        mat2: A 2D list of ints/floats
    Returns:
        A new 2D list with elements added element-wise,
        or None if mat1 and mat2 are not the same shape
    """
    if len(mat1) != len(mat2):
        return None
    if len(mat1[0]) != len(mat2[0]):
        return None
    result = []
    for i in range(len(mat1)):
        new_row = []
        for j in range(len(mat1[i])):
            new_row.append(mat1[i][j] + mat2[i][j])
        result.append(new_row)
    return result

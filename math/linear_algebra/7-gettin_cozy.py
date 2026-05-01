#!/usr/bin/env python3
"""Module for concatenating 2D matrices."""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenate two 2D matrices along a specific axis.
    Args:
        mat1: A 2D list of ints/floats
        mat2: A 2D list of ints/floats
        axis: 0 to concatenate vertically (rows)
    Returns:
        A new concatenated matrix, or None if matrices
    """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        result = []
        for i in range(len(mat1)):
            result.append(mat1[i][:] + mat2[i][:])
        return result
    return None

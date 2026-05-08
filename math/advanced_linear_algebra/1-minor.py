#!/usr/bin/env python3
"""Module to calculate the minor matrix of a matrix"""
determinant = __import__('0-determinant').determinant


def minor(matrix):
    """Calculates the minor matrix of a matrix"""
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix) or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")
    if n == 1:
        return [[1]]
    minor_matrix = []
    for i in range(n):
        row_minors = []
        for j in range(n):
            sub_matrix = [row[:j] + row[j+1:] for row in (matrix[:i] +
                          matrix[i+1:])]
            row_minors.append(determinant(sub_matrix))
        minor_matrix.append(row_minors)
    return minor_matrix

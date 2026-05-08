#!/usr/bin/env python3
"""Module to calculate the cofactor matrix of a matrix"""
minor = __import__('1-minor').minor


def cofactor(matrix):
    """Calculates the cofactor matrix of a matrix"""
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix) or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")
    if n == 1:
        return [[1]]
    m_matrix = minor(matrix)
    cofactor_matrix = []
    for i in range(n):
        row_cofactors = []
        for j in range(n):
            sign = (-1) ** (i + j)
            row_cofactors.append(m_matrix[i][j] * sign)
        cofactor_matrix.append(row_cofactors)
    return cofactor_matrix

#!/usr/bin/env python3
"""Module to calculate the inverse of a matrix"""
determinant = __import__('0-determinant').determinant
adjugate = __import__('3-adjugate').adjugate


def inverse(matrix):
    """Calculates the inverse of a matrix"""
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if n == 0 or matrix == [[]] or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Step 1: Check if the matrix is singular
    det = determinant(matrix)
    if det == 0:
        return None

    # Step 2: Handle 1x1 case directly
    if n == 1:
        return [[1 / matrix[0][0]]]

    # Step 3: Get the adjugate matrix
    adj = adjugate(matrix)

    # Step 4: Divide every element of the adjugate by the determinant
    return [[element / det for element in row] for row in adj]

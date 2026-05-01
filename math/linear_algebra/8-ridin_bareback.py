#!/usr/bin/env python3
"""Function that performs matrix multiplication"""


def mat_mul(mat1, mat2):
    """Returns the product of two 2D matrices or None if incompatible"""
    if len(mat1[0]) != len(mat2):
        return None
    result = [[sum(mat1[row][k] * mat2[k][col]
                   for k in range(len(mat2)))
               for col in range(len(mat2[0]))]
              for row in range(len(mat1))]
    return result

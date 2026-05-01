#!/usr/bin/env python3
"""Function that performs element-wise operations on numpy arrays"""


def np_elementwise(mat1, mat2):
    """Returns tuple of element-wise sum, difference, product, quotient"""
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)

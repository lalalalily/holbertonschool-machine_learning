#!/usr/bin/env python3
"""Module for summation of i squared"""


def summation_i_squared(n):
    """Calculates the sum of i^2 from 1 to n"""
    if type(n) is not int or isinstance(n, bool) or n < 1:
        return None
    return (n * (n + 1) * (2 * n + 1)) // 6

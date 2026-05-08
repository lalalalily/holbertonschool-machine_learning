#!/usr/bin/env python3
"""Module for summation of i squared"""


def summation_i_squared(n):
    """Calculates the sum of i^2 from 1 to n using the direct formula"""
    if not isinstance(n, (int, float)) or n < 0:
        return None
    if n == 0:
        return 0
    result = (n * (n + 1) * (2 * n + 1)) / 6
    return int(result)

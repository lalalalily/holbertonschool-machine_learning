#!/usr/bin/env python3
"""Module to calculate the derivative of a polynomial"""


def poly_derivative(poly):
    """Calculates the derivative of a polynomial"""
    if type(poly) is not list or len(poly) == 0:
        return None
    for c in poly:
        if type(c) not in (int, float):
            return None
    if len(poly) == 1:
        return [0]
    res = [poly[i] * i for i in range(1, len(poly))]
    while len(res) > 1 and res[-1] == 0:
        res.pop()
    return res

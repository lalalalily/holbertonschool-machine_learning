#!/usr/bin/env python3
"""Module to calculate the integral of a polynomial"""


def poly_integral(poly, C=0):
    """Calculates the integral of a polynomial"""
    if type(poly) is not list or len(poly) == 0 or type(C) not in (int, float):
        return None
    res = [int(C) if float(C).is_integer() else C]
    for i, coeff in enumerate(poly):
        if type(coeff) not in (int, float):
            return None
        val = coeff / (i + 1)
        res.append(int(val) if val.is_integer() else val)
    while len(res) > 1 and res[-1] == 0:
        res.pop()
    return res

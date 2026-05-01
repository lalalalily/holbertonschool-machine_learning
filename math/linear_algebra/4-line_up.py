#!/usr/bin/env python3
"""Module for adding arrays element-wise."""


def add_arrays(arr1, arr2):
    """
    Add two arrays element-wise.
    Args:
        arr1: A list of ints/floats
        arr2: A list of ints/floats
    Returns:
        A new list with elements added element-wise,
        or None if arr1 and arr2 are not the same shape
    """
    if len(arr1) != len(arr2):
        return None
    result = []
    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])
    return result

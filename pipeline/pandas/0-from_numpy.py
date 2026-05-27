#!/usr/bin/env python3
"""
Module to create a pandas DataFrame from a numpy ndarray.
"""
import pandas as pd


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray with columns labeled
    in alphabetical order and capitalized.
    """
    columns = [chr(65 + i) for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=columns)

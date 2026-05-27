#!/usr/bin/env python3
"""
Defines a function that converts a slice of a DataFrame to a np.ndarray
"""


def array(df):
    """
    Selects the last 10 rows of the High and Close columns from a DataFrame
    and converts them into a numpy.ndarray.

    Args:
        df: The input DataFrame.

    Returns:
        The numpy array of the selected values.
    """
    return df[['High', 'Close']].tail(10).to_numpy()

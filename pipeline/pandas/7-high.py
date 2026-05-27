#!/usr/bin/env python3
"""
Module to sort a DataFrame by a column value.
"""


def high(df):
    """
    Sorts the DataFrame by the High price in descending order.
    """
    return df.sort_values(by='High', ascending=False)

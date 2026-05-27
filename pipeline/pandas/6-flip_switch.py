#!/usr/bin/env python3
"""
Module to reverse and transpose a DataFrame.
"""


def flip_switch(df):
    """
    Sorts the data in reverse chronological order and transposes it.
    """
    return df.iloc[::-1].T

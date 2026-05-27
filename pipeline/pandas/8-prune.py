#!/usr/bin/env python3
"""
Module to clean missing values from explicit target columns.
"""


def prune(df):
    """
    Removes any entries where Close has NaN values.
    """
    return df.dropna(subset=['Close'])

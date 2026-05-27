#!/usr/bin/env python3
"""
Module to calculate clean descriptive statistics for a DataFrame.
"""


def analyze(df):
    """
    Computes descriptive statistics for all columns except the Timestamp column.
    """
    return df.drop(columns=['Timestamp']).describe()

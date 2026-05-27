#!/usr/bin/env python3
"""
Module to calculate clean descriptive statistics for a DataFrame.
"""


def analyze(df):
    """
    Computes descriptive stats for all columns except the Timestamp.
    """
    return df.drop(columns=['Timestamp']).describe()

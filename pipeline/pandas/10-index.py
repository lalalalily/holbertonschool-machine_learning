#!/usr/bin/env python3
"""
Module to alter DataFrame index mappings.
"""


def index(df):
    """
    Sets the Timestamp column as the index of the dataframe.
    """
    return df.set_index('Timestamp')

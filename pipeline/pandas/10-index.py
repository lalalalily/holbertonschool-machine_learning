#!/usr/bin/env python3
"""
Module to alter DataFrame index mappings.
"""
import pandas as pd


def index(df):
    """
    Sets the Timestamp column as the index of the dataframe.
    """
    return df.set_index('Timestamp')

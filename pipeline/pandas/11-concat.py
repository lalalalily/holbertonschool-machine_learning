#!/usr/bin/env python3
"""
Module to index and concatenate two separate financial DataFrames.
"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    Indexes dataframes, filters down rows, and merges them with multi-keys.
    """
    df1 = index(df1)
    df2 = index(df2)
    df2_filtered = df2.loc[df2.index <= 1417411920]
    return pd.concat([df2_filtered, df1], keys=['bitstamp', 'coinbase'])

#!/usr/bin/env python3
"""
Module to index and concatenate two separate financial DataFrames.
"""
import pandas as pd


def concat(df1, df2):
    """
    Indexes dataframes, filters down rows, and merges them with multi-keys.
    """
    index_func = __import__('10-index').index
    df1 = index_func(df1)
    df2 = index_func(df2)
    df2_filtered = df2.loc[:1417411920]
    return pd.concat([df2_filtered, df1], keys=['bitstamp', 'coinbase'])

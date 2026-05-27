#!/usr/bin/env python3
"""
Module to build a chronological MultiIndex hierarchy from two DataFrames.
"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Rearranges MultiIndex levels and slices by timestamp range.
    """
    df1 = index(df1)
    df2 = index(df2)
    
    df1_filtered = df1.loc[(df1.index >= 1417411980) & (df1.index <= 1417417980)]
    df2_filtered = df2.loc[(df2.index >= 1417411980) & (df2.index <= 1417417980)]
    
    df = pd.concat([df2_filtered, df1_filtered], keys=['bitstamp', 'coinbase'])
    df = df.swaplevel(0, 1)
    df = df.sort_index(level=0)
    return df

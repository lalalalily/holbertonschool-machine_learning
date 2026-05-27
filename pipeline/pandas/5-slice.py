#!/usr/bin/env python3
"""
Module to slice a DataFrame down to specific columns and step intervals.
"""
import pandas as pd


def slice(df):
    """
    Extracts the columns High, Low, Close, and Volume_(BTC)
    and selects every 60th row.
    """
    return df[['High', 'Low', 'Close', 'Volume_(BTC)']].iloc[::60]

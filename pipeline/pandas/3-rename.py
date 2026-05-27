#!/usr/bin/env python3
"""
Module to rename and convert columns in a DataFrame.
"""
import pandas as pd


def rename(df):
    """
    Renames the Timestamp column to Datetime, converts values to datetime
    format, and displays only the Datetime and Close columns.
    """
    df = df.rename(columns={'Timestamp': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    return df[['Datetime', 'Close']]

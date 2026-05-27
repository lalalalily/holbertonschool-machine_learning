#!/usr/bin/env python3
"""
Module to handle comprehensive missing data rules.
"""
import pandas as pd


def fill(df):
    """
    Cleans and backfills missing values based on custom column rules.
    """
    df.drop(columns=['Weighted_Price'], inplace=True)
    df['Close'] = df['Close'].ffill()
    df['Open'] = df['Open'].fillna(df['Close'])
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)
    return df

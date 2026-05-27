#!/usr/bin/env python3
"""
Defines a function that fills missing values in a pd.DataFrame
"""


def fill(df):
    """
    Cleans and fills missing values in a cryptocurrency DataFrame
    according to specific project rules.

    Args:
        df: The input DataFrame.

    Returns:
        The modified DataFrame with filled values.
    """
    # 1. Remove the Weighted_Price column
    df = df.drop(columns=['Weighted_Price'])

    # 2. Missing values in Close set to the previous row value (forward fill)
    df['Close'] = df['Close'].ffill()

    # 3. Missing values in High, Low, Open set to the same row's Close value
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])

    # 4. Missing values in Volume columns set to 0
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    return df

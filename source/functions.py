# Import Libraries

import pandas as pd
import numpy as np
import os
import yfinance as yf
from fredapi import Fred


# Calculate Logarithmic Returns
def log_returns(
        price_series: pd.Series
):
    return np.log(price_series / price_series.shift(1))


# Function to import data
def import_daily_financial_data(
        ticker: str,
        start_date: str = '2018-01-01',
        end_date: str = '2025-01-01',
        returns: bool = False,
):
    # Get the Data from Yahoo Finance
    data = yf.download(
        ticker,  # Stock to import
        start=start_date,  # First Date
        end=end_date,  # Last Date
        interval='1d',  # Daily Basis
        auto_adjust=True  # Adjusted Prices
    )

    # Flat columns
    data.columns = data.columns.get_level_values(0)
    data.columns = data.columns.str.lower()

    if returns:
        data['returns'] = log_returns(data['close'])

    # get rid of nans
    data.dropna(inplace=True)

    return data


# Data Collection Function from FRED
def get_fred_data(
        symbol: str,
        fred_key: str,
) -> pd.DataFrame:
    # Key to access the API
    key = fred_key

    # Access
    fred = Fred(api_key=key)

    # DataFrame
    df = fred.get_series(symbol)

    return df
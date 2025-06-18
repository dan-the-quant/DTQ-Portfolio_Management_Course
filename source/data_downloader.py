# Import Data Management
import pandas as pd
import numpy as np

# Import Providers Libraries
import yfinance as yf

# Handle Files
import sys
import os

# Import Local Functions
sys.path.append(os.path.abspath("../source"))
from config import get_data_provider


# Calculate Logarithmic Returns
def log_returns(
        price_series: pd.Series
):
    return np.log(price_series / price_series.shift(1))


# Function to import data
def import_yf_financial_data(
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


# Main Get Data Function
def get_market_data(
        ticker,
        start_date,
        end_date,
        returns: bool = False,
):
    # Set provider:
    provider = get_data_provider()

    if provider == 'yahoo_finance':
        return import_yf_financial_data(
            ticker,
            start_date,
            end_date,
            returns
        )

    else:
        raise ValueError(f"Unknown data provider: {provider}")

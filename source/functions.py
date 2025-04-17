# Import Libraries

import pandas as pd
import numpy as np
from fredapi import Fred


# Function to import data
def import_financial_data(
        ticker: str,
        starting_year: str = '2015',
        adjusted_close: bool = True,
        volume: bool = True,
        mkt_cap: bool = False,
):
    # Check the ticker for Upper Cases
    ticker = ticker if ticker.isupper() else ticker.upper()

    # Import data
    df = pd.read_csv(rf"..\stocks\{ticker}.csv")

    # Set the Index
    date_col = 'Date' if 'Date' in df.columns else 'date' if 'date' in df.columns else None
    if date_col:
        df = df.set_index(date_col)
        df.index = pd.to_datetime(df.index)

    columns = [
        'Open Price',
        'High Price',
        'Low Price',
        'Close Price',
    ]

    rename_dict = {
        "Open Price": "open",
        "High Price": "high",
        "Low Price": "low",
        "Close Price": "close",
    }

    if adjusted_close:
        columns.append('Adjusted_close')
        rename_dict['Adjusted_close'] = 'adj_close'

    if volume:
        columns.append('Volume')
        rename_dict["Volume"] = "volume"

    if mkt_cap:
        columns.append('Company Market Cap')
        rename_dict["Company Market Cap"] = "mkt_cap"

    df_useful_data = df[columns]
    df_useful_data = df_useful_data.rename(columns=rename_dict)

    return df_useful_data.loc[f"{starting_year}-01-01":]


# Data Collection Function from FRED
def get_fred_data(
        symbol: str,
        fred_key: str,
) -> pd.DataFrame:

    key = fred_key

    fred = Fred(api_key=key)

    df = fred.get_series(symbol)

    return df
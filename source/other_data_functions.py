# Import Libraries
import pandas as pd
from fredapi import Fred

# Handle Files
import sys
import os

# Import Local Functions
sys.path.append(os.path.abspath("../source"))
from config import get_api_key


# Data Collection Function from FRED
def get_fred_data(
        symbol: str,
) -> pd.DataFrame:
    # Key to access the API
    key = get_api_key(provider='fred')

    # Access
    fred = Fred(api_key=key)

    # DataFrame
    df = fred.get_series(symbol)

    return df

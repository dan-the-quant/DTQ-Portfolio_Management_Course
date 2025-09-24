# Libraries
import pandas as pd
import numpy as np


# Analytics
def calculate_analytics(
        returns_dataframe,
        risk_free_rate=0.0
):
    # To calculate in percentage
    returns_dataframe = returns_dataframe * 100

    # Trading Days in one Year
    ann_factor = 252

    # Annualized Returns
    annualized_return = returns_dataframe.mean() * ann_factor

    # Annualized Volatility
    annualized_std = returns_dataframe.std() * np.sqrt(ann_factor)

    # Sharpe Ratio
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_std

    # Max Drawdown
    cumulative_returns = np.exp(returns_dataframe.div(100).cumsum())            # cumulative returns from log returns
    rolling_max = cumulative_returns.rolling(ann_factor, min_periods=1).max()   # rolling maximum
    drawdown = cumulative_returns / rolling_max - 1                             # drawdown
    max_drawdown = drawdown.min() * 100                                         # maximum drawdown

    # VaR at 95%
    var_95 = returns_dataframe.quantile(0.05)

    # Create DF
    summary_df = pd.DataFrame({
        "Annualized Returns": annualized_return,
        "Annualized Volatility": annualized_std,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "VaR 95%": var_95
    })

    return summary_df

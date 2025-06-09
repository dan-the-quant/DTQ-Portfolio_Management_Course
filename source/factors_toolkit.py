import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from typing import Union


# Create the Weights Function
def wexp(N, half_life):
    c = np.log(0.5) / half_life
    n = np.array(range(N))
    w = np.exp(c * n)
    return np.flip(w / np.sum(w))


# Let us Create a new function
def FamaFrenchFactors(
        stock_returns: pd.Series,
        market_returns: pd.Series,
        small_minus_big_series: pd.Series,
        high_minus_low_series: pd.Series,
):
    # Align time series to the same date range
    common_index = stock_returns.index.intersection(market_returns.index) \
        .intersection(small_minus_big_series.index) \
        .intersection(high_minus_low_series.index)

    stock_returns = stock_returns.loc[common_index]
    market_returns = market_returns.loc[common_index]
    small_minus_big_series = small_minus_big_series.loc[common_index]
    high_minus_low_series = high_minus_low_series.loc[common_index]

    X = pd.concat([market_returns, small_minus_big_series, high_minus_low_series], axis=1)
    y = stock_returns

    # Create weights with exponential decay
    T = len(y)
    weights = T * wexp(T, T / 2)

    # Fit WLS regression
    model = sm.WLS(y, sm.add_constant(X), weights=weights, missing='drop').fit()

    # Avoid KeyError by checking if params exist
    params = model.params

    alpha = params.iloc[0]
    capm_beta = params.iloc[1]
    smb_beta = params.iloc[2]
    hml_beta = params.iloc[3]

    parameters = {
        'alpha': alpha,
        'mkt_beta': capm_beta,
        'smb_beta': smb_beta,
        'hml_beta': hml_beta,
    }

    return parameters


def newey_west_std(
        errors,
        lag=4
):
    T = len(errors)
    gamma_var = errors.var()  # Start with variance of the series

    for l in range(1, lag + 1):
        weight = 1 - (l / (lag + 1))
        autocov = np.cov(errors[:-l], errors[l:])[0, 1]  # Autocovariance at lag l
        gamma_var += 2 * weight * autocov  # Newey-West adjustment

    return np.sqrt(gamma_var / T)  # Standard error


def fama_macbeth_significance_test(
        gamma_series,
        lag=4
):
    gamma_means = gamma_series.mean()

    # Compute Newey-West adjusted standard errors
    gamma_std = gamma_series.apply(newey_west_std, lag=lag)

    # Compute t-statistics
    t_stats = gamma_means / gamma_std

    # Compute p-values
    p_values = 2 * (1 - stats.t.cdf(abs(t_stats), df=len(gamma_series) - 1))

    # Create results DataFrame
    results = pd.DataFrame({
        'Mean Gamma': gamma_means,
        'Std Error': gamma_std,
        't-stat': t_stats,
        'p-value': p_values
    })

    return results

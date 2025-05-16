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


# Create a Function that Calculates the Market Cap Weighted Average
def market_cap_weighted_average(
        var_df: pd.DataFrame,
        mkt_cap_df: pd.DataFrame,
        mask: pd.DataFrame,
) -> pd.Series:
    # Exclude stocks filtering by mask
    mkt_cap_adj = mkt_cap_df.fillna(0) * mask

    # Calculate the Weighted Mean
    numerator = (var_df * np.sqrt(mkt_cap_adj)).sum(axis=1)
    denominator = np.sqrt(mkt_cap_adj).sum(axis=1)

    return numerator / denominator


# Create a Function that Calculates the Standard Deviation
def standard_deviation(
        var_df: pd.DataFrame,
):
    return var_df.std(axis=1, ddof=1)


# Standardize Function
def standardize_zscore(
        var_df: pd.DataFrame,
        mkt_cap_df: pd.DataFrame,
        mask: pd.DataFrame,
) -> pd.DataFrame:
    # Calculate Market Cap Weighted Average
    wa = market_cap_weighted_average(var_df, mkt_cap_df, mask)

    # Calculate Cross-Sectional Standard Deviation
    std = standard_deviation(var_df)

    # Standardize (broadcasting Series across DataFrame rows)
    zscore_df = (var_df.subtract(wa, axis=0)).divide(std, axis=0)

    return zscore_df


# Winsorize Function
def custom_winsorize(df):
    df_winz = df.copy()
    data = df_winz.to_numpy()

    # Masks for each condition
    mask_gt_10 = data > 10
    mask_lt_minus10 = data < -10
    mask_5_to_10 = (data > 5) & (data <= 10)
    mask_minus10_to_minus5 = (data >= -10) & (data < -5)

    # Apply transformations
    data[mask_5_to_10] = 5
    data[mask_minus10_to_minus5] = -5
    data[mask_gt_10 | mask_lt_minus10] = np.nan

    # Return as DataFrame
    return pd.DataFrame(data, index=df.index, columns=df.columns)


# Standardization
def iterative_standardize_winsorize(
        var_df: pd.DataFrame,
        mkt_cap_df: pd.DataFrame,
        mask: pd.DataFrame,
        iterations: int = 3
) -> pd.DataFrame:
    result = var_df.copy()

    for i in range(iterations):
        result = standardize_zscore(result, mkt_cap_df, mask)
        result = custom_winsorize(result)

    # Last standardization
    result = standardize_zscore(result, mkt_cap_df, mask)

    return result


# Helper: Common Index
def align_with_common_index(
        target: pd.Series,
        *others: Union[pd.Series, pd.DataFrame]
):
    # Set the target index
    common_index = target.index

    # Find the common intersections
    for obj in others:
        common_index = common_index.intersection(obj.index)

    # Reindex
    target_aligned = target.loc[common_index]
    aligned_others = [obj.loc[common_index] for obj in others]

    return target_aligned, *aligned_others


# Calculate the Factor Betas
def estimate_factor_betas(
        stock_returns: pd.Series,
        factor_returns: pd.DataFrame,
        half_life: float = None,
) -> dict:
    # Align indices
    stock_returns, factor_returns = align_with_common_index(
        stock_returns, factor_returns
    )

    # Ensure data is valid
    if len(stock_returns) < 5 or factor_returns.empty:
        return {}

    T = len(stock_returns)
    hl = half_life if half_life else T / 2
    weights = T * wexp(T, hl)

    X = sm.add_constant(factor_returns)
    y = stock_returns

    model = sm.WLS(y, X, weights=weights, missing='drop').fit()
    params = model.params.to_dict()

    # Rename intercept to alpha
    if 'const' in params:
        params['alpha'] = params.pop('const')

    return params


# Calculate the Rolling Factor Betas
def rolling_factor_betas(
        stock_returns: pd.Series,
        factor_returns: pd.DataFrame,
        window: int = 252,
        half_life: float = None,
) -> pd.DataFrame:
    # Align index
    stock_returns, factor_returns = align_with_common_index(
        stock_returns, factor_returns
    )

    # Combine data
    data = pd.concat([stock_returns, factor_returns], axis=1).dropna()
    results = []

    for end in range(window, len(data) + 1):
        window_data = data.iloc[end - window:end]
        y = window_data.iloc[:, 0]
        X = window_data.iloc[:, 1:]

        # Estimate betas using modular function
        betas = estimate_factor_betas(y, X, half_life=half_life)

        # Only append if regression succeeded
        if betas:
            betas['date'] = data.index[end - 1]
            results.append(betas)

    result_df = pd.DataFrame(results).set_index('date')

    return result_df


def fama_macbeth_regression(
        returns_df: pd.DataFrame,
        mkt_cap_df: pd.DataFrame,
        *betas_dfs: pd.DataFrame
) -> pd.DataFrame:
    # Intersect common dates
    common_dates = returns_df.index.intersection(mkt_cap_df.index)
    for betas_df in betas_dfs:
        common_dates = common_dates.intersection(betas_df.index)

    # Intersect assets available for all
    common_assets = returns_df.columns.intersection(mkt_cap_df.columns)
    for betas_df in betas_dfs:
        common_assets = common_assets.intersection(betas_df.columns)

    # List for storing results
    betas_list = []

    # Loop through each common date
    for date in common_dates:

        # Data for this date
        returns = returns_df.loc[date].dropna()
        mkt_caps = mkt_cap_df.loc[date].dropna()

        # Combine betas for all factors for this date
        betas_combined = pd.DataFrame(index=common_assets)
        for betas_df in betas_dfs:
            betas_combined = pd.concat([betas_combined, betas_df.loc[date].dropna()], axis=1)

        # Keep only the common assets between returns, mkt_caps, and betas_combined
        y = returns.loc[common_assets]
        X = betas_combined.loc[common_assets]

        # Calculate weights (normalized market cap)
        weights = np.sqrt(mkt_caps.loc[common_assets])
        weights = weights / weights.sum()

        # Add constant term to the factors
        X = sm.add_constant(X)

        # Run the regression
        model = sm.WLS(y, X, weights=weights).fit()
        params = model.params
        params.name = date
        betas_list.append(params)

    # Combine the results into a DataFrame
    history_betas_df = pd.DataFrame(betas_list)
    return history_betas_df


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

import pandas as pd
import numpy as np
import statsmodels.api as sm


# Create the Weights Function
def wexp(N, half_life):
    c = np.log(0.5) / half_life
    n = np.array(range(N))
    w = np.exp(c * n)
    return np.flip(w / np.sum(w))


# Weighted Moving Averages
def weighted_moving_average(data, window_size):
    # Exponential Weights
    weights = window_size * wexp(window_size, window_size / 2)

    return data.rolling(window=window_size).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


# Helper: exclude tiny returns
def n_days_nonmiss(
        returns,
        tiny_ret=1e-6
):
    ix_ret_tiny = np.abs(returns) <= tiny_ret
    return np.sum(~ix_ret_tiny, axis=0)


# Relative Strength Calculation
def calc_rstr(
        returns,
        half_life=126,
        min_obs=100,
):
    rstr = np.log(1. + returns)
    if half_life == 0:
        weights = np.ones_like(rstr)
    else:
        weights = len(returns) * np.asmatrix(wexp(len(returns), half_life)).T
    rstr = np.sum(rstr * weights)
    idx = n_days_nonmiss(returns) < min_obs
    rstr.where(~idx, other=np.nan, inplace=True)
    df = pd.Series(rstr)
    df.name = returns.index[-1]
    return df


# Rolling Relative Strength
def rolling_calc_rstr(
        returns,
        window_size=252,
        half_life=126,
        min_obs=100
):
    rolling_results = []
    range_to_iter = range(len(returns) - window_size + 1)
    for i in range_to_iter:
        window_returns = returns.iloc[i:i + window_size]
        rs_i = calc_rstr(
            returns=window_returns,
            half_life=half_life,
            min_obs=min_obs
        )

        rolling_results.append(rs_i)

    return pd.concat(rolling_results, axis=1)


# Compute Daily Interest Rate
def annual_to_daily_rate(
        series,
        days_per_year=252
):
    # From % to decimal
    decimal_series = series / 100

    # Calculate the Rate
    return (1 + decimal_series) ** (1 / days_per_year) - 1


# Create the CAPM Function
def capm_regression(
        excess_stock: pd.Series,
        excess_benchmark: pd.Series,
        window: int = 252,
        WLS: bool = False,
):
    X = excess_benchmark
    y = excess_stock

    if WLS:
        # Create weights with exponential decay
        weights = window * wexp(window, window / 2)

        # Fit WLS regression
        model = sm.WLS(y, sm.add_constant(X), weights=weights, missing='drop').fit()

    else:
        # Fit OLS regression
        model = sm.OLS(y, sm.add_constant(X), missing='drop').fit()

    return model


def rolling_capm_regression(
        stock_returns: pd.Series,
        benchmark_returns: pd.Series,
        daily_rfr: pd.Series,
        window: int = 252,
        WLS: bool = False,
):
    # Align Data
    df = pd.concat([stock_returns, benchmark_returns, daily_rfr], axis=1)
    df = df.dropna()
    df.columns = ['stock_returns', 'benchmark_returns', 'daily_returns']

    # Compute Excess Returns
    excess_stock = df['stock_returns'] - df['daily_returns']
    excess_benchmark = df['benchmark_returns'] - df['daily_returns']

    # Lists
    alphas, betas, sigma = [], [], []
    dates = []

    for t in range(window, len(excess_stock)):
        # The variables
        X = excess_benchmark.iloc[t - window:t]
        y = excess_stock.iloc[t - window:t]

        # Create the Model
        model = capm_regression(y, X, window=window, WLS=WLS)

        # Avoid KeyError by checking if params exist
        params = model.params
        r_sigma = model.resid.std()

        # Append values
        alphas.append(params.iloc[0])
        betas.append(params.iloc[1])
        sigma.append(r_sigma)
        dates.append(excess_stock.index[t - 1])  # Last date to calculate betas

    parameters = pd.DataFrame({
        'alpha': alphas,
        'beta': betas,
        'sigma': sigma,
    }, index=pd.Index(dates, name="date"))

    return parameters


# Compute the Factor Contribution to Returns
def compute_factor_contributions(
        factor_returns: pd.DataFrame,
        betas: pd.DataFrame
):
    # Multiply Elements
    if isinstance(factor_returns, pd.Series):
        contribution = (factor_returns * betas)
    elif isinstance(factor_returns, pd.DataFrame):
        contribution = (factor_returns * betas).sum(axis=1)
    else:
        contribution = None

    return contribution


# Compute the Residual Returns
def compute_residual_returns(
        stock_excess_returns: pd.Series,
        factor_returns: pd.DataFrame,
        betas: pd.DataFrame
):
    # Multiply Elements
    contribution = compute_factor_contributions(factor_returns, betas)

    return stock_excess_returns - contribution

import pandas as pd
import numpy as np


# Portfolio Variance Function
def portfolio_variance(
        weights,
        returns
):
    # Weights
    weights = np.array(weights)

    # Calculate Covariance Matrix
    cov_matrix = returns.cov()

    # Portfolio Variance
    port_var = weights.T @ cov_matrix @ weights

    return port_var


# Rolling Portfolio Variance Function
def rolling_portfolio_variance(
        returns_df,
        weights,
        window=252
):
    weights = np.array(weights)

    rolling_vars = []
    index = returns_df.index

    for i in range(window - 1, len(returns_df)):
        window_returns = returns_df.iloc[i - window + 1: i + 1]
        cov_matrix = np.cov(window_returns.T)
        var = weights.T @ cov_matrix @ weights
        rolling_vars.append(var)

    result = pd.Series([np.nan] * (window - 1) + rolling_vars, index=index)
    return result


# Helper: Preprocess inputs
def _preprocess_mkwz_inputs(
        expected_returns,
        cov_matrix
):
    # Process Inputs
    mu = expected_returns.values.flatten().reshape(-1, 1)
    Sigma = cov_matrix.values
    Sigma_inv = np.linalg.inv(Sigma)
    iota = np.ones_like(mu)

    return mu, Sigma, Sigma_inv, iota


# Helper: Compute Efficient Frontier components
def eff_components(
        expected_returns,
        cov_matrix
):
    # Get Inputs
    mu, _, Sigma_inv, iota = _preprocess_mkwz_inputs(expected_returns, cov_matrix)

    # Calculate components
    A = mu.T @ Sigma_inv @ mu
    B = iota.T @ Sigma_inv @ mu
    C = iota.T @ Sigma_inv @ iota
    D = (A * C) - (B * B)
    return A, B, C, D


# Helper: Get the Efficient Frontier Coefficients
def eff_coefficients(
        expected_returns,
        cov_matrix
):
    # Get the components
    A, B, C, D = eff_components(expected_returns, cov_matrix)

    # Calculate the coefficients
    pi_0 = A / D
    pi_1 = 2 * B / D
    pi_2 = C / D

    return pi_0.item(), pi_1.item(), pi_2.item()


# Function to get the efficient frontier
def eff_equation(
        coefficients,
        desired_returns
):
    # Get the coefficients
    pi_0, pi_1, pi_2 = coefficients

    # Set the desired returns
    mu_P = desired_returns

    return np.sqrt(pi_0 - pi_1 * mu_P + pi_2 * mu_P ** 2)


# Function to get the Markowitz's Optimization Weights
def markowitz_weights(
        expected_returns,
        cov_matrix,
        desired_returns
):
    # Inputs
    mu, _, Sigma_inv, iota = _preprocess_mkwz_inputs(expected_returns, cov_matrix)

    # Components
    A, B, C, D = eff_components(expected_returns, cov_matrix)

    # Calculate weights
    w_1 = ((desired_returns * C - B) / D) * (Sigma_inv @ mu)
    w_2 = ((A - desired_returns * B) / D) * (Sigma_inv @ iota)

    return (w_1 + w_2).flatten()


def rolling_markowitz_weights(
        returns,
        desired_returns,
        window=252,
        rebalance_freq=126
):
    # Lists to Store Things
    weights_list = []
    dates = []

    for i in range(window, len(returns), rebalance_freq):
        # Prepare Inputs
        past_returns = returns.iloc[i - window:i]  # Rolling Window
        past_excepted_returns = past_returns.mean()
        past_cov_matrix = past_returns.cov()

        # Calculate Weights
        w = markowitz_weights(past_excepted_returns, past_cov_matrix, desired_returns)

        # Save weights and dates
        weights_list.append(w)
        dates.append(returns.index[i])

    # Create the DataFrame
    weights_df = pd.DataFrame(weights_list, index=dates, columns=returns.columns)

    # Expand the DataFrame
    weights_df = weights_df.reindex(returns.index, method='ffill')

    return weights_df.dropna()


# Function to get the CAL Optimization Weights
def cal_weights(
        rfr,
        expected_returns,
        cov_matrix,
        desired_returns
):
    # Inputs
    mu, _, Sigma_inv, _ = _preprocess_mkwz_inputs(expected_returns, cov_matrix)

    # Calculate the A component
    A = mu.T @ Sigma_inv @ mu

    # Calculate weights
    weights = ((desired_returns - rfr) / A) * (Sigma_inv @ mu)

    return weights


# Function to get the CAL volatility for a desired level of returns
def cal_volatility(
        rfr,
        expected_returns,
        desired_returns,
        cov_matrix
):
    # Inputs
    mu, _, Sigma_inv, _ = _preprocess_mkwz_inputs(expected_returns, cov_matrix)

    # Calculate the A component
    A = mu.T @ Sigma_inv @ mu

    return abs(np.sqrt(1 / A) * (desired_returns - rfr))


# Function to get the CAL returns for a desired level of volatility
def cal_equation(
        rfr,
        sharpe_ratio,
        sigma_P
):
    return rfr + sharpe_ratio * sigma_P


# Analytics
def calculate_analytics(
        df_returns,
        risk_free_rate=0.0
):
    # Trading Days in one Year
    ann_factor = 252

    # Annualized Returns
    annualized_return = df_returns.mean() * ann_factor

    # Annualized Volatility
    annualized_std = df_returns.std() * np.sqrt(ann_factor)

    # Sharpe Ratio
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_std

    # Max Drawdown
    cumulative_returns = (1 + df_returns.div(100)).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / rolling_max) - 1
    max_drawdown = drawdown.min()

    # VaR at 95%
    var_95 = df_returns.quantile(0.05)

    # Create DF
    summary_df = pd.DataFrame({
        "Annualized Returns": annualized_return,
        "Annualized Volatility": annualized_std,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "VaR 95%": var_95
    })

    return summary_df


# Helper: Preprocess inputs
def _preprocess_zb_inputs(
        expected_betas,
        cov_matrix
):
    # Process Inputs
    beta = expected_betas.values.flatten().reshape(-1, 1)
    Sigma = cov_matrix.values
    Sigma_inv = np.linalg.inv(Sigma)
    iota = np.ones_like(beta)

    return beta, Sigma, Sigma_inv, iota


# Helper: Compute Zero Beta Portfolio components
def zb_components(
        expected_betas,
        cov_matrix
):
    # Get Inputs
    beta, _, Sigma_inv, iota = _preprocess_zb_inputs(expected_betas, cov_matrix)

    # Calculate components
    C = np.dot(np.dot(iota.T, Sigma_inv), iota)
    D = np.dot(np.dot(beta.T, Sigma_inv), beta)
    E = np.dot(np.dot(beta.T, Sigma_inv), iota)
    Delta = (D * C - E * E)

    return C, D, E, Delta


# Function to get the Zero Beta Optimization Weights
def zero_beta_weights(
        expected_betas,
        cov_matrix,
):
    # Inputs
    beta, _, Sigma_inv, iota = _preprocess_zb_inputs(expected_betas, cov_matrix)

    # Components
    C, D, E, Delta = zb_components(expected_betas, cov_matrix)

    # Calculate weights
    beta_weights = ((D / Delta) * (Sigma_inv @ iota)) - ((E / Delta) * (Sigma_inv @ beta))

    return beta_weights.flatten()


def rolling_zero_beta_weights(
        returns,
        betas,
        window=252,
        rebalance_freq=63
):
    # Lists
    weights_list = []
    dates = []

    for i in range(window, len(returns), rebalance_freq):
        # Prepare Inputs
        past_returns = returns.iloc[i - window:i]  # Rolling Window
        past_betas = betas.iloc[i - window:i]
        past_excepted_betas = past_betas.mean()
        past_cov_matrix = past_returns.cov()

        # Calculate Weights
        w = zero_beta_weights(past_excepted_betas, past_cov_matrix)

        # Save weights and dates
        weights_list.append(w)
        dates.append(returns.index[i])

    # Create the DataFrame
    weights_df = pd.DataFrame(weights_list, index=dates, columns=returns.columns)

    # Expand the DataFrame
    weights_df = weights_df.reindex(returns.index, method='ffill')

    return weights_df.dropna()

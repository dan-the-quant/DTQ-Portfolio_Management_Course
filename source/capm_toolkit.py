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


# Detrending Series Function
def detrending_series(
        y: pd.Series(),
        residuals=True
):
    Y = y.dropna()

    trend = pd.Series(
        np.arange(1, len(Y) + 1),
        index=Y.index
    )

    models = [
        sm.OLS(Y, sm.add_constant(np.ones_like(Y))),
        sm.OLS(Y, sm.add_constant(trend)),
        sm.OLS(Y, sm.add_constant(pd.DataFrame({"trend": trend, "trend_sq": trend ** 2}))),
        sm.OLS(Y, sm.add_constant(pd.DataFrame({"trend": trend, "trend_sq": trend ** 2, "trend_cb": trend ** 3}))),
        sm.OLS(Y, sm.add_constant(
            pd.DataFrame({"trend": trend, "trend_sq": trend ** 2, "trend_cb": trend ** 3, "trend_qua": trend ** 4}))),
    ]

    results = [model.fit() for model in models]
    aics = [result.aic for result in results]

    best_model_index = np.argmin(aics)
    best_result = results[best_model_index]

    # print(best_result.summary())

    if residuals:
        return best_result.resid

    else:
        return best_result.fittedvalues

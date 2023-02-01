#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import norm


#%%
def decisive_factor(sigma, S, x_value, ntime, interest, dt_zero=False):
    """Function tocalculate call price for strike kappa/s

    Parameters:
    sigma: float, volatility
    S: float, asset price
    x_value: float, portfolio value
    ntime: int, number of timesteps
    interest: float, interest rate
    dt_zero: bool, whether to use the formula for dt=0

    Returns:
    call price (float)"""

    if x_value == 0 or dt_zero:
        return S * sigma
    d1 = (
        (np.log(1 + np.abs(x_value) / S) + (sigma**2 / 2 + interest) / ntime)
        / sigma
        * np.sqrt(ntime)
    )
    d2 = d1 - sigma / np.sqrt(ntime)
    return (S + np.abs(x_value)) * norm.cdf(d1) - S * norm.cdf(d2) * np.exp(
        -interest / ntime
    )


# %%


def decisive_factor_vec(x_value, ntime, interest, dt_zero=False):
    """Create a vectorized function for calculating call prices for an array of asset values,
    given a portfolio vlue x and interest rate interest

    Parameters:
    x_value: float, portfolio value
    ntime: int, number of timesteps
    interest: float, interest rate
    dt_zero: bool, whether to use the formula for dt=0

    Returns:
    vectorized function for calculating call price (np.vectorize)"""

    f_temp = np.vectorize(
        lambda sig, S: decisive_factor(
            sigma=sig,
            S=S,
            x_value=x_value,
            ntime=ntime,
            interest=interest,
            dt_zero=dt_zero,
        )
    )
    return f_temp


def check_hypothesis(p, X, S, market_env):
    """Check whether actions p are optimal for arrays of portfolio values, assets

    Parameters:
    p: list of np.arrays, actions
    X: np.array, portfolio values
    S: np.array, asset values
    market_env: FinMa, market environment

    Returns:
    list of percentage of correct actions at each timestep"""
    d = market_env.dim
    Sigma = np.sqrt(np.diag(market_env.stock.sigma))

    AP_correct = []
    for t in range(market_env.ntime):
        inds = d * np.greater(X[:, t], 0)
        for i, x in enumerate(X[:, t]):
            factors = decisive_factor_vec(x, market_env.ntime, market_env.interest)(
                Sigma, S[i, t, :]
            )
            inds[i] = inds[i] + np.argmax(factors)

        pred = np.argmax(p[t], axis=-1)
        correct_pred = np.mean(np.equal(pred, inds))
        AP_correct.append(correct_pred)
        print(f"{correct_pred*100:.2f}% correct actions at time {t}")

    return AP_correct

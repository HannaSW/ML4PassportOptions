#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import torch

# %%


def create_paths(
    filepath,
    npath,
    ntime,
    Sigma,
    mu,
    r,
    s0,
    retBM=False,
    save_paths=True,
):
    """Create paths of underlying assets.

    Parameters
    ----------
    filepath : str
        Path to save the paths.
    npath : int
        Number of paths.
    ntime : int
        Number of time steps.
    Sigma: 2d list
        covariance of BM
    mu: 1d list
        mean of BM
    r: float
        interest rate
    s0: 1d list
        initial values of assets
    retBM: bool, optional
        Return BM paths along with asset paths. The default is False.
    save_paths: bool, optional
        Save paths. The default is True.

    Returns
    -------
    S: 3d np.array
        Paths of underlying assets.

    """
    d = len(mu)  # dimension of random variables
    C = np.linalg.cholesky(Sigma)  # Sigma = CC^T
    W_incr = np.random.multivariate_normal(
        mean=mu, cov=np.diag(np.ones(d)), size=[npath, ntime]
    ) / np.sqrt(ntime)
    drift = (np.repeat(r, d) - np.diagonal(Sigma) * 0.5) / ntime

    # realisations of stocks in dimenstions npath x ntime x d
    S = np.multiply(
        np.exp(
            np.cumsum(
                drift + W_incr.dot(C.T),
                axis=1,
            )
        ),
        s0[:, None, :],
    )

    # add s0 to all paths
    S = np.concatenate((s0[:, None, :], S), axis=1)

    if save_paths:
        # save training data
        np.save(filepath + "/S_" + str(d), S)
        np.save(filepath + "/W_" + str(d), W_incr)

    if retBM:
        return (W_incr, S)
    else:
        return S


# %%


def portfolio_value(strat, stock, interest, x0):
    """Calculate portfolio values for some strategy, stocks and interest rate.

    Args:
        strat: strategies, np.array of dimension (npath, ntime, dim)
        stock: stock values, np.array of dimension (npath, ntime+1, dim)
        interest: interest rate, float
        x0: initial portfolio value, np.array of dimension (npath,1)

    Returns:
        portfolio values at timepoints (1,..., ntime) for every path, np array of
        dimension (npath, ntime)
    """

    npath = stock.shape[0]
    ntime = stock.shape[1] - 1
    x = np.concatenate(
        (x0, np.zeros((npath, ntime))),
        axis=1,
    )
    stock_old = stock[:, 0, :]

    # loop over timepoints 0:(ntime-1)
    for i in range(ntime):
        stock_gain = np.reshape(
            np.sum(
                (stock[:, i + 1, :] - stock_old) * strat[:, i, :],
                axis=-1,
            ),
            (npath, 1),
        )
        int_gain = np.reshape(
            np.multiply(
                (np.exp(interest / ntime) - 1),
                x[:, i]
                - np.sum(
                    stock_old * strat[:, i, :],
                    axis=-1,
                ),
            ),
            (npath, 1),
        )
        x[:, (i + 1)] = np.squeeze(
            np.expand_dims(x[:, i], axis=1) + stock_gain + int_gain
        )
        stock_old = stock[:, i + 1, :]
    return x


#%%


def calc_state(
    model,
    poss_action,
    d,
    x,
    s,
    r,
    ntime,
    t_start,
    t_end,
    hardmax=False,
    market_env=None,
):
    """Function to calculate portfolio state at a given time, given initial values,
    asset paths, and a model for strategies.

    Parameters:
    model: torch.nn.Sequential or torch.nn.Module, model for NN-actions
    poss_action: list,  possible trading actions
    d: int, dimension of underlying
    x: np.array, dim=(npath,1) initial portfolio value
    s: np.array, dim=(npath, t_end-t_start, d) stock values
    r: float, interest rate
    ntime: int, number of timesteps
    t_start: int, start time
    t_end: int, end time
    hardmax: boolean, should harmax activation be used for strategies?
    market_env: FinMa, environment for market

    Returns:
    portfolio value at time t_start-t_end for given asset paths (np.array of shape (npath, 1))"""

    npath = x.shape[0]
    num_actions = len(poss_action)

    for k in range(t_end):
        tensor_input = torch.tensor(
            np.concatenate(
                (np.ones((npath, 1)) * (t_start + k) / ntime, x, s[:, k, :]),
                axis=-1,
            ),
            dtype=torch.float,
        )
        if market_env is not None:
            tensor_input = market_env.create_additional_states(tensor_input)

        if isinstance(model, list):
            pi = (model[k](tensor_input).detach().numpy()[:, :d]).reshape((npath, 1, d))
            if hardmax:
                pi = pi / np.linalg.norm(pi, axis=-1, keepdims=True)
        else:
            probs = model(tensor_input).detach().numpy()[:, :num_actions]
            if hardmax:
                p_max = np.argmax(probs, axis=-1)

                pi = np.reshape(
                    list(
                        map(
                            lambda x: poss_action[x],
                            p_max,
                        )
                    ),
                    (npath, 1, d),
                )
            else:
                pi = np.reshape(
                    list(
                        map(
                            lambda x: poss_action[np.random.choice(2 * d, p=x)],
                            probs,
                        )
                    ),
                    (npath, 1, d),
                )

        x = portfolio_value(strat=pi, stock=s[:, k : (k + 2), :], interest=r, x0=x)[
            :, 1, None
        ]

    return x


# %%
def calc_price(
    model,
    asset_env,
    x0,
    s0,
    npath,
    poss_action,
    ntime,
    r,
    hardmax=False,
    market_env=None,
    ignore_weights=False,
    weights=None,
):
    """Function to calculate price (discounted) of option for initial state and a trained strategy model.

    Parameters:
    model: torch.nn.Sequential or torch.nn.Module, model for NN-actions
    asset_env: BSassets, environment for assets
    x0: float, initial portfolio value
    s0: list, initial asset values
    npath: int, number of MC paths to be used in calculation of price
    poss_action: list,  possible trading actions
    ntime: int, number of timesteps
    r: float, interest rate
    hardmax: boolean, should harmax activation be used for strategies?
    market_env: MarketEnvironment, environment for market, if None, no additional states are added
    ignore_weights: boolean, if True, weights are ignored and only the mean of positive part is returned
    weights: list, weights for positive part and absolute value

    Returns:
    portfolio value at time t_start-t_end for given asset paths (np.array of shape (npath, 1))"""

    x0path = np.ones((npath, 1)) * x0
    spath = asset_env.create_paths(
        npaths=npath,
        ntimesteps=ntime,
        initial_val=s0 * np.ones((npath, 1, asset_env.dim)),
    )[0]
    xT = calc_state(
        model,
        poss_action,
        asset_env.dim,
        x0path,
        spath,
        r,
        ntime,
        t_start=0,
        t_end=ntime,
        hardmax=hardmax,
        market_env=market_env,
    )

    if not ignore_weights:
        factor = 1 / (weights[0] + 2 * weights[1])
        value = weights[0] * np.mean(np.maximum(xT, 0)) + weights[1] * np.mean(
            np.abs(xT)
        )
        return value * factor * np.exp(-r) + x0 * weights[1] * factor
    return np.mean(np.maximum(xT, 0)) * np.exp(-r)


# %%


def strategies(
    npath,
    ntime,
    d,
    model,
    S,
    r,
    poss_action,
    hardmax,
    x0path,
    market_env=None,
):
    """Function to calculate strategies for given asset paths and a trained strategy model.

    Parameters:
    npath: int, number of MC paths for which strategies are calculated
    ntime: int, number of timesteps
    d: int, dimension of assets
    model: torch.nn.Module or list of torch.nn.Module or torch.nn.Sequential, model for NN-actions
    S: np.array, dim=(npath, t_end-t_start, d) stock values
    r: float, interest rate
    poss_action: list,  possible actions for strategies
    hardmax: boolean, should harmax activation be used for strategies?
    x0path: np.array, dim=(npath,1) initial portfolio value
    market_env: FinMa, environment for market, if None, no additional states are added

    Returns:
    array of strategies for given asset paths (np.array of shape (npath, 1))
    list of probabilities for each timestep (list of np.arrays of shape (npath, len(poss_action))"""

    # determine approximated strategy
    q = np.zeros((npath, ntime, d))
    probs = []
    xpath = x0path

    for k in range(ntime):
        tensor_input = torch.tensor(
            np.concatenate(
                (np.ones((npath, 1)) * k / ntime, xpath, S[:, k, :]),
                axis=-1,
            ),
            dtype=torch.float,
        )
        if market_env is not None:
            tensor_input = market_env.create_additional_states(tensor_input)

        if isinstance(model, list):
            q[:, k, :] = model[k](tensor_input).detach().numpy()[:, :d]
        else:
            probs.append(model(tensor_input).detach().numpy()[:, : len(poss_action)])

            if hardmax:
                p_max = np.argmax(probs[-1], axis=-1)

                q[:, k, :] = list(
                    map(
                        lambda x: poss_action[x],
                        p_max,
                    )
                )
            else:
                q[:, k, :] = list(
                    map(
                        lambda x: poss_action[np.random.choice(2 * d, p=x)],
                        probs[-1],
                    )
                )

        xpath = portfolio_value(
            q[:, [k], :],
            S[:, k : (k + 2), :],
            r,
            xpath,
        )[:, 1, None]

    return (q, probs)

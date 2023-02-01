#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch

from hypotheses import decisive_factor_vec


def simple_agent(market_env, random=False):
    """Function to create agents taking constant or random actions.

    Parameters:
    market_env: FinMa, market environment object
    random: boolean, random actions?


    Returns:
    function that takes time&state input and returns actions"""

    a_range = market_env.dim * 2

    def actions(tensor_input):
        npath = tensor_input.shape[0]
        a = np.zeros((npath, a_range))
        if random:
            inds = np.random.choice(a=a_range, size=npath)
        else:
            inds = np.repeat(0, npath)

        a[[i for i in range(npath)], inds] = 1
        at = torch.tensor(
            a,
            dtype=torch.float,
        )
        return at

    return actions


def hypo_agent(market_env, dt_zero=False):
    """Function to create agents taking the optimal action -sign(x) in the asset with highest call price

    Parameters:
    market_env: FinMa, market environment object
    dt_zero: bool, whether to use the formula for dt=0


    Returns:
    function that takes time&state input and returns actions"""

    d = market_env.dim
    a_range = d * 2

    def actions(tensor_input):
        npath = tensor_input.shape[0]
        a = np.zeros((npath, a_range))
        if d == 1:
            inds = 1 * np.greater(tensor_input[:, 1], 0)
        else:
            Sigma = np.sqrt(np.diag(market_env.stock.sigma))
            inds = d * np.greater(tensor_input[:, 1], 0)
            for i, x in enumerate(tensor_input[:, 1]):
                factors = decisive_factor_vec(
                    x_value=x,
                    ntime=market_env.ntime,
                    interest=market_env.interest,
                    dt_zero=dt_zero,
                )(Sigma, tensor_input[i, 2 : 2 + d])
                inds[i] += np.argmax(factors)
        a[[i for i in range(npath)], inds] = 1
        at = torch.tensor(
            a,
            dtype=torch.float,
        )
        return at

    return actions

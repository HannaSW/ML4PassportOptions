#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

CONFIG_NO = 2
SEED = 1

SAVE = {"model": True, "plots": True, "plot_path": None}

# market params
MARKET_PARAMS = {
    "interest": 0.002,  # interest rate
    "ntime": 10,  # number of timesteps per path
    "x0": 0,  # initial portfolio value
    "d": 2,  # dimension of asset
    "s0": [1, 1],  # initial values for assets
    "mu": [0, 0],  # mean of BM
    "sig": 0.04,  # var of BM
    "cor": np.eye(2),  # correlation matrix
}

# Model Parameters
A2C_PARAMS = {
    "npath": 100,
    "MTnetwork": False,
}
A2C_TRAIN_PARAMS = {
    "actor_lr": 0.0001,
    "critic_lr": 0.0007,
    "tau": 1e-11,
    "max_iter": 400000,
    "weight_decay": 0,
    "nsteps_newPV": None,
    "nsteps_newS": None,
    "weights": [0, 1],
}
PG_PARAMS = {
    "npath": 2 ** 13,  # number of paths to be simulated
    "max_t_minus": 9,
    "weights": [0, 1],
    "hardmax": False,
}
PG_TRAIN_PARAMS = {
    "lr": 0.001,
    "batch_size": 2 ** 4,
    "epochs": 2 ** 6,
    "l2": 1e-18,
    "entropy_reg": 1e-11,
}


# evaluation parameters
EVAL_PARAMS = {
    "x0": 0,
    "use_wandb": False,
    "select_best_A2C": False,
    "ntime": 32,
    "filename_A2C": None,
    "filename_PG": None,
    "filename_DH": None,
}

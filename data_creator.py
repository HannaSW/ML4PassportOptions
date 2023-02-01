import numpy as np
import market_functions as mf
import copy
from datetime import datetime
from util import timediff_d_h_m_s, custom_sign

#%%
def calc_reward(
    terminal_value,
    weights=[1, 0],
):
    """Function to calculate reward for samples of terminal value, given a weighting of MC estimates.

    Parameters:
    terminal_value: np.array, realized terminal portfolio values
    weights: list,  weights for MC estimates in this order: positive part, absolute value

    Returns:
    reward: np.array, reward for each sample"""
    r_pp = np.maximum(terminal_value, 0)
    r_abs = np.abs(terminal_value)
    return weights[0] * r_pp + weights[1] * r_abs


#%%
def data_creator(
    model,
    asset_env,
    r,
    x0,
    ntime,
    npath,
    poss_action,
    t_minus=0,
    verbose=False,
    hardmax=False,
    weights=[1, 0],
    market_env=None,
):
    """Function to create a data point (state, action)-pair at timepoint ntime-t_minus-1
    for a NN that maps into probabilities of optimal actions (relaxed target).

    Parameters:
    model: torch.nn.Module or torch.nn.Sequential, current strategy network
    asset_env: FinMa, asset environment object
    r: float, interest rate
    x0: float, initial portfolio value
    ntime: int, number of timesteps in market
    npath: int, number of MC paths
    poss_action: list of length 2*d, possible trading actions
    t_minus: int, data is created for time point ntime-t_minus
    verbose: boolean, print time for data creation
    hardmax: boolean, should harmax activation be used for strategies in forward pass?
    weights: list of length 2, weights for MC estimates in terminal reward: w[0] positive part, w[1] absolute value
    market_env: market environment, if None, no additional states are assumed in addition to time, portfolio value and asset values

    Returns:
    tuple of training input and target"""

    # start to take time ----------------------------------------------------------
    if verbose:
        start = datetime.now()
    # -----------------------------------------------------------------------------

    t = ntime - t_minus - 1

    terminal_reward = 0.0  # expected terminal reward
    while terminal_reward == 0.0:

        # sample one asset path up to t to get typical state (x,s) at time t
        ##### ALTERTNATIVE: could sample from N(.,.) and log(N(.,.)) ------START

        if t == 0:
            s_sample = asset_env.initial_state[0, :, :]  # s0* np.ones((1, d))
        else:
            s_sample = asset_env.create_paths(
                npaths=1,
                ntimesteps=t,
                initial_val=asset_env.initial_state[
                    0:1, :, :
                ],  # s0 * np.ones((1, 1, d)),
            )[0]
        # sample initial portfolio value
        x_start = x0 + np.random.normal(size=1, scale=0.1)
        xpath = x_start * np.ones((1, 1))
        xpath = mf.calc_state(
            model,
            poss_action,
            asset_env.dim,
            x=xpath,
            s=s_sample,
            r=r,
            ntime=ntime,
            t_start=0,
            t_end=t,
            hardmax=hardmax,
            market_env=market_env,
        )

        ##### ALTERTNATIVE: could sample from N(.,.) and log(N(.,.))   ------END

        # determine state value at t, and create input for network to be
        # trained at timepoint t
        st_sample = s_sample[:, -1, :]
        xt = copy.deepcopy(xpath[0, 0])
        xtpath = np.expand_dims(np.repeat(xt, npath), axis=1)
        input_t = (
            [np.ones((1, 1)) * xt]
            + [np.ones((1, asset_env.dim)) * st_sample]
            + [np.ones((1, 1)) * t / ntime]
        )

        ##### Now MC-approx of the reward of each action starts to obtain a label
        ##### for our input state
        # sample npath asset paths up to maturity
        s_sample = asset_env.create_paths(
            npaths=npath,
            ntimesteps=ntime - t,
            initial_val=st_sample * np.ones((npath, 1, asset_env.dim)),
        )[0]
        # determine target q
        for q in poss_action:
            # x_t+1
            xpath = mf.portfolio_value(
                q * np.ones((npath, 1, asset_env.dim)),
                s_sample[:, 0:2, :],
                r,
                xtpath,
            )[:, 1, None]

            # x_T using iteratively NN-outputs as strategies
            xpath = mf.calc_state(
                model,
                poss_action,
                asset_env.dim,
                x=xpath,
                s=s_sample[:, 1:, :],
                r=r,
                ntime=ntime,
                t_start=t + 1,
                t_end=ntime - t - 1,
                hardmax=hardmax,
                market_env=market_env,
            )

            # terminal reward. option to set a different target to reduce bias for positive portfolio values
            if asset_env.mu == 0:
                print("attention: resetting weights to [1,0]!")
                weights = [0, 1]

            rewards = calc_reward(terminal_value=xpath, weights=weights)
            reward = np.mean(rewards)
            if reward > terminal_reward:  # terminal rewards of the best action
                terminal_reward = reward
                q_max = q

    # determine target y_true
    y = np.zeros((1, 2 * asset_env.dim))  ##### NEEDED, should be outside the while loop
    y[:, poss_action.index(q_max)] = 1

    # add additional tasks to target
    if model.add_tasks:
        for at in model.add_tasks:
            if at not in ["sign", "volas"]:
                raise NotImplementedError("additional task not implemented")
            else:
                if at == "sign":
                    y = np.concatenate((y, np.ones((1, 1)) * custom_sign(xt)), axis=1)
                elif at == "volas":
                    y = np.concatenate(
                        (
                            y,
                            np.sqrt(np.diag(asset_env.sigma)).reshape(1, asset_env.dim),
                        ),
                        axis=1,
                    )

    # print time difference -------------------------------------------------------
    if verbose:
        end = datetime.now()
        diff = end - start
        print(
            "Elapsed: {}d {}h:{}m:{}s".format(*timediff_d_h_m_s(diff)),
            "(" + datetime.now().strftime("%H:%M %d-%m-%Y") + ")",
        )
    # -----------------------------------------------------------------------------
    return input_t, y

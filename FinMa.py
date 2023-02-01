#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
# add gym-directory to path
# ----------------------------------------------------------------------------
import numpy as np

try:
    import gym
except ModuleNotFoundError:
    import sys

    pathtogym = input("please input the directory of your gym installation:")
    sys.path.append(pathtogym)
    import gym

from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from datetime import datetime
from torch.distributions.categorical import Categorical
import torch

from hypotheses import decisive_factor_vec
from util import custom_sign
from data_creator import calc_reward

# %%
# -----------------------------------------------------------------------------
# Define Financial Market environment
# ----------------------------------------------------------------------------


class BSassets:
    """implements Black Scholes environment"""

    def __init__(
        self,
        interest,
        s0,
        sigma,
        mu,
        dim,
        ntime,
        npath=1,
    ):

        # set asset parameters
        self.dim = dim
        self.ntime = ntime
        self.npath = npath
        self.interest = interest
        self.sigma = sigma  # variance of BM
        self.mu = mu  # mean of BM
        self.stock_drift = (
            np.repeat(interest, self.dim) - np.diagonal(sigma) * 0.5
        ) / ntime

        self.initial_state = np.ones((npath, 1, dim)) * s0
        self.state = np.ones((npath, 1, dim)) * s0
        self.ret = np.zeros((npath, 1, dim))
        self.log_ret = np.ones((npath, 1, dim))

    def step(self, keep_env_pathdim=False):
        """perform a step in the asset environment"""
        new_state, log_ret = self.create_paths(
            ntimesteps=1, keep_env_pathdim=keep_env_pathdim
        )

        self.log_ret = log_ret
        self.ret = new_state[:, [1], :] - new_state[:, [0], :]
        self.state = new_state[:, [1], :]

    def reset(self, new_initial=False, **kwargs):
        """reset financial asset value and returns

        Parameters
        ----------
        new_inital: bool, should new initial value be sampled?"""

        if new_initial:
            s0 = np.exp(
                np.random.multivariate_normal(
                    mean=self.mu,
                    cov=self.sigma,
                )
            )  # / np.sqrt(self.ntime))#np.random.lognormal(size=self.dim)
            self.initial_state = (
                np.ones(
                    (
                        self.npath,
                        1,
                        self.dim,
                    )
                )
                * s0
            )
        self.state = np.copy(self.initial_state)
        self.ret = np.zeros((self.npath, 1, self.dim))
        self.log_ret = np.ones((self.npath, 1, self.dim))

    def create_paths(
        self,
        ntimesteps,
        npaths=None,
        keep_env_pathdim=False,
        initial_val=None,
    ):
        """creates npath paths over ntimesteps time steps of the stock within the BS env.
        returns asset paths an log returns

        Parameters
        ----------
        ntimesteps: int, number of timesteps for which to sample asset values
        npaths: int, number of paths to sample
        keep_env_pathdim: bool, should the number of paths be kept the same as in the environment?
        initial_val: np.array, initial value of the asset. if not given, the initial value of the environment is used.
        """

        if keep_env_pathdim:
            npaths = self.npath
        elif npaths is None:
            npaths = 1

        if initial_val is None:
            if npaths > self.npath:
                npaths = self.npath
                print(
                    f"no initial value given. taking initial value of environment. this means the number of paths is {npaths}."
                )

            initial_val = self.state[0:npaths, :, :]

        W = np.random.multivariate_normal(
            mean=self.mu,
            cov=self.sigma,
            size=[npaths, ntimesteps],
        ) / np.sqrt(self.ntime)
        # realisations of stocks in dimenstions npath x ntime x dim
        S = np.multiply(
            np.exp(
                np.cumsum(
                    self.stock_drift + W,
                    axis=1,
                )
            ),
            initial_val,
        )

        # prepend s0 to all paths
        S = np.concatenate(
            (
                initial_val,
                S,
            ),
            axis=1,
        )
        return (S, W)


class FinMa(gym.Env):
    """Financial market environment"""

    def __init__(
        self,
        pv0,
        ntime,
        interest,
        dim,
        asset_model,
        npath=1,
        seed=None,
        weights=[1, 0],
        additional_states=None,
    ):
        # set action space (possible values for trading strategy)
        self.dim = dim
        self.action_space = spaces.Discrete(2 * dim)
        self.weights = weights

        poss_action = np.concatenate((np.eye(dim), -np.eye(dim))).tolist()

        dictionary = dict((a, b) for a, b in enumerate(poss_action))

        self.dict = dictionary
        self.poss_action = poss_action

        # set market parameters
        self.initial_pv = np.ones((npath, 1, 1)) * pv0  # initial portfolio value
        self.ntime = ntime  # number of (trading)timesteps
        self.npath = npath  # number of paths
        self.interest = interest  # interest rate
        self.stock = asset_model

        if additional_states is not None:
            self.additional_state_dict = {}
            for s in additional_states:
                if s not in ["ret", "log_ret", "Ssigma", "signX"]:
                    raise NotImplementedError(f"additional state {s} not implemented")
                else:
                    if "ret" == s:
                        self.additional_state_dict["ret"] = self.stock.ret
                    if "log_ret" == s:
                        self.additional_state_dict["log_ret"] = self.stock.log_ret
                    if "Ssigma" == s:
                        self.additional_state_dict["Ssigma"] = (
                            np.sqrt(np.diag(self.stock.sigma)) * self.stock.state
                        )
                    if "signX" == s:
                        self.additional_state_dict["signX"] = custom_sign(
                            self.initial_pv
                        )

        else:
            self.additional_state_dict = None

        self.reset(resetpaths=True)
        self.seed(seed)

    def seed(self, seed=None):
        """seed the environment"""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def update_additional_state(self):
        """update additional state"""
        for s in self.additional_state_dict.keys():
            if s not in ["ret", "log_ret", "Ssigma", "signX"]:
                raise NotImplementedError(f"additional state {s} not implemented")
            else:
                if "ret" == s:
                    self.additional_state_dict["ret"] = self.stock.ret
                if "log_ret" == s:
                    self.additional_state_dict["log_ret"] = self.stock.log_ret
                if "Ssigma" == s:
                    self.additional_state_dict["Ssigma"] = (
                        np.sqrt(np.diag(self.stock.sigma)) * self.stock.state
                    )
                if "signX" == s:
                    self.additional_state_dict["signX"] = custom_sign(self.pv)

    def set_state(
        self,
    ):
        """set current state of market environment (time, pv, assets)"""

        base_state = np.concatenate(
            (
                np.ones(
                    (
                        self.npath,
                        1,
                    )
                )
                * self.time_run
                / self.ntime,
                self.pv.reshape(
                    (
                        self.npath,
                        1,
                    )
                ),
                self.stock.state.reshape(
                    (
                        self.npath,
                        self.dim,
                    )
                ),
            ),
            axis=-1,
        )
        if self.additional_state_dict is not None:
            self.update_additional_state()
            for key, value in self.additional_state_dict.items():
                base_state = np.concatenate(
                    (
                        base_state,
                        value.reshape(
                            (
                                self.npath,
                                -1,
                            )
                        ),
                    ),
                    axis=-1,
                )
        self.state = base_state  # copy.deepcopy(base_state)

    def step(
        self,
        action,
        log_type=0,
        keep_env_pathdim=False,
    ):
        """
        function to perform one step given an action
        if paths are provided, new stock is selected from these paths
        if no paths are provided to the object, new stock is sampled

        Parameters
        ----------
        action : np.array
            action to be performed
        log_type : int
            flag for logger
        keep_env_pathdim : bool
            whether to use the number of paths of the environment

        Returns
        -------
        state : np.array
            new state of the environment
        reward : float
            reward for the action
        done : bool
            whether the episode is done
        """

        self.time_run += 1  # timepoint

        if self.paths is None:
            self.stock.step(keep_env_pathdim=keep_env_pathdim)
            ret = self.stock.ret
        else:
            stock_new = self.paths[self.path_run, self.time_run, :]
            ret = stock_new - self.stock.state

        # set new state & stock
        stock_gain = np.sum(
            ret * action,
            axis=-1,
            keepdims=True,
        )
        int_gain = (
            self.interest
            * (
                self.pv
                - np.sum(
                    np.multiply(self.stock.state, action),
                    axis=-1,
                    keepdims=True,
                )
            )
            / self.ntime
        )

        self.pv += stock_gain + int_gain

        self.set_state()

        # determine reward
        done = bool(self.time_run == self.ntime)  # reward only at terminal time
        reward = 0
        if done:
            reward = calc_reward(
                terminal_value=self.pv,
                weights=self.weights,
            )

        # logger
        if log_type == 0:
            print("time {}:".format(self.time_run))
            print("action: {}".format(action))
            print("pv: {}".format(self.pv))
            print("reward: {}".format(reward))
            print("done: {}".format(done))
        elif log_type == 1 and done:
            print("time {}:".format(self.time_run))
            print("action: {}".format(action))
            print("pv: {}".format(self.pv))
            print("reward: {}".format(reward))
            print("done: {}".format(done))

        return self.state, reward, done

    def reset(
        self,
        resetpaths=False,
        new_initial_stock=False,
        new_initial_pv=False,
        new_paths=False,
    ):
        """
        function to reset financial market environment
        portfolio value, stock (asset values), time_run and state are always
        set to initial values

        Parameters
        ----------
        resetpaths: bool, if True paths, BM, and path_run are reset
        new_initial_stock: bool, if True new initial values for stocks are sampled
        new_initial_pv: bool, if True new initial portfolio value is sampled
        new_paths: bool, if True new indices for paths are sampled in historic data env

        """

        if new_initial_pv:
            self.initial_pv = np.ones((self.npath, 1, 1)) * np.random.normal(
                0,
                0.1,
            )

        self.pv = np.copy(self.initial_pv)
        self.time_run = 0
        self.stock.reset(new_initial=new_initial_stock, new_paths=new_paths)
        self.set_state()

        if resetpaths:
            self.paths = None
            self.BM = None
            self.path_run = 0

    def sample_action(
        self,
        t=None,
        jumptimes=False,
    ):
        """sample action

        Parameters
        ----------
        t : int
            timepoint
        jumptimes : bool
            whether to sample actions according to jumptimes

        Returns
        -------
        action : np.array
            sampled action"""
        if jumptimes:
            # 2)  sample random actions based on jumptimes
            njumps = np.random.geometric(0.7)
            jumps = np.random.geometric(0.2, size=njumps)
            if t == 0:
                pos = self.action_space.sample()
                action = self.poss_action[pos]
            elif t in jumps:
                tmp = self.poss_action.copy()
                tmp.remove(action)
                pos = self.action_space.sample()
                action = self.poss_action[pos]
        else:
            # 1) sample uniformly from action space
            k = self.action_space.sample()
            action = self.poss_action[k]

        return action

    def obtain_nn_action(
        self,
        actor,
        state,
    ):
        """take action according to actor

        Parameters
        ----------
        actor : nn.Module
            actor network
        state : np.array
            current state of the environment

        Returns
        -------
        action : np.array
            action taken by actor
        """

        # given current state, the actor outputs two values as probability for each action
        state = torch.tensor(
            state,
            dtype=torch.float,
        )
        prob = actor(state)

        if actor.add_tasks is not None:
            prob = prob[:, : len(self.poss_action)]

        # create the distribution as current policy
        pi = Categorical(prob)
        # sample one action from this policy
        a = pi.sample()

        # interact with the environment
        action = self.poss_action[a]
        return action

    def create_training_data(
        self,
        npath,
        reward_requirement,
        relaxed=True,
        log_type=2,
        plot=False,
        jumptimes=False,
        artist="random",
        foldername="",
    ):
        """
        creates training data set based on random strategies that exceed a
        reward requirement

        (note: if paths are provided to the object, one such good random
        strategy is determined for each of these paths)
        """

        training_data = []  # stores states (time, pv, asset) of accepted "game"
        training_targets = []  # stores targets of accepted "game"
        accepted_scores = []  # stores rewards that exceed threshold

        count = 0.0  # counts number of games played

        while len(accepted_scores) < npath:
            count += 1
            game_memory = []  # temporarily stores state and strat for game

            for t in range(self.ntime):
                # set new time point, portfolio value and stock
                previous_observation = self.state
                action = self.sample_action(
                    t=t,
                    jumptimes=jumptimes,
                )

                # collect time point, portfolio value, stock and strategy in game memory
                game_memory.append([previous_observation, action])

                # perform step
                state, reward, done = self.step(
                    action,
                    log_type=log_type,
                )

            # collect terminal time point, portfolio value, stock in game memory
            game_memory.append([self.state, None])

            # provide plot for showing development of randomly generated paths
            if plot:
                self.create_movie(
                    game_memory,
                    artist=artist,
                    reward_requirement=reward_requirement,
                    name=foldername,
                )

            # if strategy was successful append to dataset
            if reward > reward_requirement:

                accepted_scores.append(reward)
                for data in game_memory[:-1]:
                    # append strategy
                    if relaxed:  # relaxed target
                        target = data[1]
                        if self.dim > 1:
                            target = list(target)
                        ind = list(self.dict.keys())[
                            list(self.dict.values()).index(target)
                        ]
                        output = np.zeros((2 * self.dim,))
                        output[ind] = 1
                    else:  # exact strategy as target
                        output = data[1]

                    training_data.append(data[0])
                    training_targets.append(output)

                self.path_run += 1  # increase path index

                # logger
                if log_type == 2:
                    print("number accepted: {}".format(len(accepted_scores)))
                    print("terminal action: {}".format(action))
                    print("pv: {}".format(self.state[0, 1]))

            self.reset(new_paths=True)  # reset state

        self.path_run = 0  # reset path index
        print("number of games played: ")
        print(count)
        print("ratio of random expert games: ")
        print(npath / count)

        return training_data, training_targets

    def create_movie(
        self,
        game_memory,
        artist="HSW",
        reward_requirement=None,
        name="",
    ):
        """creates movie of a game

        Parameters
        ----------
        game_memory : list
            list of states and actions
        artist : str
            name of artist
        reward_requirement : float
            reward threshold
        name : str
            name of folder

        Returns
        -------
        None"""

        FFMpegWriter = manimation.writers["ffmpeg"]
        metadata = dict(
            title="Portfolio path",
            artist=artist,
            comment="A trade",
        )
        writer = FFMpegWriter(fps=5, metadata=metadata)

        fig, ax = plt.subplots(1, 2)
        ax[0].hlines(
            [0],
            color=["r"],
            xmin=0,
            xmax=self.ntime,
        )
        ax[0].set_title("portfolio value")
        if reward_requirement is not None:
            ax[0].hlines(
                [reward_requirement, -reward_requirement],
                color=["grey", "grey"],
                xmin=0,
                xmax=self.ntime,
            )
        ax[1].set_title("asset preference")
        ax[1].hlines(
            [1],
            color=["r", "k"],
            xmin=0,
            xmax=self.ntime,
        )
        ax[1].set_title("asset")
        timestamp = datetime.now().strftime("%H:%M %d-%m-%Y")
        s = game_memory[0][0]
        a = game_memory[0][1]
        i = 1
        with writer.saving(
            fig,
            name + "Trade_" + artist + timestamp + ".mp4",
            dpi=256,
        ):
            mappable = ax[0].collections[0]
            mappable.set_clim(0, self.dim - 1)
            cmap = mappable.to_rgba([i for i in range(self.dim)])
            if self.dim > 1:
                fig.colorbar(
                    mappable,
                    orientation="horizontal",
                    ax=ax[0],
                    ticks=[i for i in range(self.dim)],
                )
            fig.tight_layout(pad=0, w_pad=0, h_pad=0)

            while i < len(game_memory):
                j = i % (self.ntime + 1)
                sn = game_memory[i][0]

                ax[0].plot(
                    (j - 1, j),
                    (s[0, 1], sn[0, 1]),
                    linewidth=1,
                    color="black",
                )

                for k in range(self.dim):
                    ax[1].plot(
                        (j - 1, j),
                        (s[0, 2 + k], sn[0, 2 + k]),
                        linewidth=1,
                        color="grey",
                    )
                if self.dim == 1 and a is not None:
                    if a[0] > 0:
                        marker = 6
                        c = "red"
                    else:
                        marker = 7
                        c = "green"
                    ax[0].plot(
                        j - 1,
                        s[0, 1],
                        color=c,
                        marker=marker,
                        linewidth=1,
                    )
                else:
                    asset_index = np.argmax(np.abs(a))
                    marker = 6 if a[asset_index] > 0 else 7
                    ax[0].plot(
                        j - 1,
                        s[0, 1],
                        color=cmap[asset_index],
                        marker=marker,
                        markersize=10,
                        linewidth=1,
                        mec="black",
                    )
                    ax[1].plot(
                        (j - 1, j),
                        (s[0, 2 + asset_index], sn[0, 2 + asset_index]),
                        linewidth=1,
                        color="black",
                    )
                    try:
                        sdom = np.argmax(
                            decisive_factor_vec(
                                x_value=s[0, 1],
                                ntime=s[0, 0],
                                interest=self.interest,
                            )(
                                np.sqrt(np.diagonal(self.stock.sigma)),
                                s[0, 2 : 2 + self.dim],
                            )
                        )
                        ax[1].plot(j - 1, s[0, 2 + sdom], "ro")
                    except AttributeError:
                        pass

                if j == self.ntime and i + 1 < len(game_memory):
                    s = game_memory[i + 1][0]
                    a = game_memory[i + 1][1]
                    i += 2
                    ax[1].clear()
                else:
                    s = sn
                    a = game_memory[i][1]
                    i += 1
                writer.grab_frame()

    def trade_with_agent(
        self,
        actor,
        npath,
        plot=True,
        log_type=2,
        artist="NN agent",
        foldername="",
    ):

        """
        creates portfolio trajectories based on NN strategies

        Parameters
        ----------
        actor : nn.Module or nn.Sequential
            NN model that provides strategies
        npath : int
            number of paths to be generated
        plot : bool, optional
            if True, a movie is created, by default True
        log_type : int, optional
            type of logging to be used, by default 2
        artist : str, optional
            name of the artist, by default "NN agent"
        foldername : str, optional
            name of the folder, by default ""

        Returns
        -------
        None

        """

        game_memory = []  # temporarily stores state and strat for game

        for count in range(npath):
            self.reset(new_paths=True)

            for t in range(self.ntime):
                # set new time point, portfolio value and stock
                previous_observation = self.state
                action = self.obtain_nn_action(
                    actor,
                    previous_observation,
                )

                # collect time point, portfolio value, stock and strategy in game memory
                game_memory.append([previous_observation, action])

                # perform step
                state, reward, done = self.step(
                    action,
                    log_type=log_type,
                )

            # collect terminal time point, portfolio value, stock in game memory
            game_memory.append([self.state, None])

        # provide plot for showing development of randomly generated paths
        if plot:
            self.create_movie(
                game_memory,
                artist=artist,
                name=foldername,
            )

    def create_additional_states(self, tensor_base_state):
        """
        creates extended tensors states for given set of base states, and additional states according to the environment

        Parameters
        ----------
        tensor_base_state: tensor of base states

        Returns
        -------
        tensor of extended states

        """
        if self.additional_state_dict is None:
            return tensor_base_state
        else:
            for key, value in self.additional_state_dict.items():
                if key == "ret":
                    tensor_base_state = torch.cat(
                        (
                            tensor_base_state,
                            torch.cat(
                                (
                                    torch.zeros((1, self.dim)),
                                    torch.diff(
                                        tensor_base_state[:, 2 : 2 + self.dim], dim=0
                                    ),
                                ),
                                dim=0,
                            ),
                        ),
                        dim=-1,
                    )
                if key == "log_ret":
                    tensor_base_state = torch.cat(
                        (
                            tensor_base_state,
                            torch.cat(
                                (
                                    torch.ones((1, self.dim)),
                                    torch.diff(
                                        torch.log(
                                            tensor_base_state[:, 2 : 2 + self.dim]
                                        ),
                                        dim=0,
                                    ),
                                ),
                                dim=0,
                            ),
                        ),
                        dim=-1,
                    )
                if key == "Ssigma":
                    tensor_base_state = torch.cat(
                        (
                            tensor_base_state,
                            tensor_base_state[:, 2 : 2 + self.dim]
                            * torch.sqrt(
                                torch.diagonal(torch.tensor(self.stock.sigma))
                            ),
                        ),
                        dim=-1,
                    )
                if key == "signX":
                    tensor_base_state = torch.cat(
                        (
                            tensor_base_state,
                            custom_sign(tensor_base_state[:, 1:2]),
                        ),
                        dim=-1,
                    )
        return tensor_base_state

    # used in deep hedging
    def calc_reward(
        self,
        terminal_value,
        weights=[1, 0],
        x_start=0,
    ):
        """Function to calculate reward for samples of terminal value, given a weighting of MC estimates.

        Parameters:
        terminal_value: tensor, realized terminal portfolio values
        weights: list,  weights for MC estimates in this order: positive part, absolute value

        Returns:
        reward (tensor)"""
        r_pp = torch.mean(torch.maximum(terminal_value - x_start, torch.zeros(1)))
        r_abs = torch.mean(torch.abs(terminal_value - x_start))
        return r_pp * weights[0] + r_abs * weights[1]

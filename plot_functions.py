#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.animation as manimation
from scipy.stats import t
import torch

# %%


def strategy_video(
    ntime,
    model,
    end,
    d=1,
    start=0,
    smin=0,
    smax=3,
    s1_value=1.25,
    s2_value=1.5,
    xmin=-1,
    xmax=1,
    x_value=0.5,
    resolution=100,
    market_env=None,
):
    """Creates a sequence of images of the strategy for a given model
     over a grid of asset and portfolio values.

    Parameters
    ----------
    ntime : int
        The number of time steps in the environment.
    model : torch.nn.Module or torch.nn.Sequential or list of torch.nn.Module
        The model to be used.
    end : int
        The last time step to be considered.
    d : int, optional
        The dimension of the stock prices. The default is 1.
    start : int, optional
        The first time step to be considered. The default is 0.
    smin : float, optional
        The minimum value of the stock prices. The default is 0.
    smax : float, optional
        The maximum value of the stock prices. The default is 3.
    s1_value : float, optional
        The fixed value of the first stock price. The default is 1.25.
    s2_value : float, optional
        The fixed value of the second stock price. The default is 1.5.
    xmin : float, optional
        The minimum value of portfolio value. The default is -1.
    xmax : float, optional
        The maximum value of portfolio value. The default is 1.
    x_value : float, optional
        The fixed value of the decision variable. The default is 0.5.
    resolution : int, optional
        The resolution of the grid. The default is 100.
    market_env : FinMa, optional
        The market environment. The default is None.

    Returns
    -------
    None.
    """
    x = np.linspace(xmin, xmax, resolution)
    s = np.linspace(smin, smax, resolution)

    if d == 1:
        xgrid, sgrid = np.meshgrid(x, s)
        xgrid = np.reshape(xgrid, (resolution**2, 1))
        sgrid = np.reshape(sgrid, (resolution**2, 1))

    elif d == 2:
        xgrid, sgrid1, sgrid2 = np.meshgrid(x, s, s)
        xgrid = np.reshape(xgrid, (resolution**3, 1))
        sgrid1 = np.reshape(sgrid1, (resolution**3, 1))
        sgrid2 = np.reshape(sgrid2, (resolution**3, 1))
        sgrid = np.concatenate([sgrid1, sgrid2], axis=1)  # 2-d stock prices

        # Extracting the right indices
        indices_sgrid = np.zeros(resolution**2, dtype=np.int32)
        for i in range(resolution):
            for j in range(resolution):
                indices_sgrid[i * resolution + j] = i * resolution**2 + j

        # Building grids for the specific values of X, S1, S2
        specific_xgrid = np.full((resolution**2, 1), x_value)

        specific_s1grid = np.full((resolution**2, 1), s1_value)
        specific_s1grid = np.concatenate(
            [
                specific_s1grid,
                np.reshape(sgrid[indices_sgrid, 1], (resolution**2, 1)),
            ],
            axis=1,
        )

        specific_s2grid = np.full((resolution**2, 1), s2_value)
        specific_s2grid = np.concatenate(
            [
                np.reshape(sgrid[indices_sgrid, 1], (resolution**2, 1)),
                specific_s2grid,
            ],
            axis=1,
        )

    else:
        print(f"no grid for dimension {d} implemented yet")

    for time_point in range(start, end):

        if d == 1:
            t = np.ones_like(xgrid) * time_point / ntime
            tensor_input = torch.tensor(
                np.concatenate((t, xgrid, sgrid), axis=-1),
                dtype=torch.float,
            )
            if market_env is not None:
                tensor_input = market_env.create_additional_states(tensor_input)

            if isinstance(model, list):
                strategy_t = model[time_point](tensor_input).detach().numpy()[:, :d]
            else:
                strategy_t = model(tensor_input).detach().numpy()[:, : 2 * d]
            plt.scatter(
                xgrid,
                sgrid,
                c=strategy_t[:, 0],
                marker=".",
                vmin=0,
                vmax=1,
            )
            plt.colorbar()
            plt.xlabel("wealth $X$")
            plt.ylabel("price $S$")
            plt.title(
                f"Prob of action 1 at timestep {time_point}"
                + " (t={:.{}f})".format(
                    time_point / ntime,
                    3,
                )
            )

        else:
            t = np.ones_like(specific_xgrid) * time_point / ntime
            fig, axs = plt.subplots(d + 1, 2 * d)
            fig.set_size_inches(18.5, 10.5)
            fig.suptitle(
                f"timepoint {time_point}"
                + " (t={:.{}f})".format(
                    time_point / ntime,
                    3,
                )
            )

            # Plot s1 against x
            axs[0, 0].set_ylabel("$S^1$")
            tensor_input = torch.tensor(
                np.concatenate(
                    (t, xgrid[0 : resolution**2], specific_s2grid),
                    axis=-1,
                ),
                dtype=torch.float,
            )
            if market_env is not None:
                tensor_input = market_env.create_additional_states(tensor_input)
            if isinstance(model, list):
                strategy_t = model[time_point](tensor_input).detach().numpy()[:, :d]
            else:
                strategy_t = model(tensor_input).detach().numpy()[:, : 2 * d]

            for j in range(2 * d):
                action = 1 if j < d else -1
                asset = (j % d) + 1
                axs[0, j].scatter(
                    xgrid[0 : resolution**2],
                    specific_s2grid[:, 0],
                    c=strategy_t[:, j],
                    marker=".",
                    vmin=0,
                    vmax=1,
                )
                axs[0, j].title.set_text(f"action {action} in asset {asset} ")
                axs[0, j].set_xlabel("$X$")

            mappable = axs[0, 0].collections[0]
            fig.colorbar(mappable, ax=axs[0, 3])

            # Plot s2 against x
            axs[1, 0].set_ylabel("$S^2$")
            tensor_input = torch.tensor(
                np.concatenate(
                    (t, xgrid[0 : resolution**2], specific_s1grid),
                    axis=-1,
                ),
                dtype=torch.float,
            )
            if market_env is not None:
                tensor_input = market_env.create_additional_states(tensor_input)

            strategy_t = model(tensor_input).detach().numpy()[:, : 2 * d]

            for j in range(2 * d):
                action = 1 if j < d else -1
                asset = (j % d) + 1
                axs[1, j].scatter(
                    xgrid[0 : resolution**2],
                    specific_s1grid[:, 1],
                    c=strategy_t[:, j],
                    marker=".",
                    vmin=0,
                    vmax=1,
                )
                axs[1, j].title.set_text(f"action {action} in asset {asset} ")
                axs[1, j].set_xlabel("$X$")

            mappable = axs[1, 0].collections[0]
            fig.colorbar(mappable, ax=axs[1, 3])

            # Plot of s2 against s1
            axs[d, 0].set_ylabel("$S^2$")
            tensor_input = torch.tensor(
                np.concatenate(
                    (t, specific_xgrid, sgrid[indices_sgrid, :]),
                    axis=-1,
                ),
                dtype=torch.float,
            )

            if market_env is not None:
                tensor_input = market_env.create_additional_states(tensor_input)
            if isinstance(model, list):
                strategy_t = model[time_point](tensor_input).detach().numpy()[:, :d]
            else:
                strategy_t = model(tensor_input).detach().numpy()[:, : 2 * d]

            for j in range(2 * d):
                action = 1 if j < d else -1
                asset = (j % d) + 1
                axs[d, j].scatter(
                    sgrid[indices_sgrid, 0],
                    sgrid[indices_sgrid, 1],
                    c=strategy_t[:, j],
                    marker=".",
                    vmin=0,
                    vmax=1,
                )
                axs[d, j].title.set_text(f"action {action} in asset {asset} ")
                axs[d, j].set_xlabel("$S^1$")

            mappable = axs[d, 0].collections[0]
            fig.colorbar(mappable, ax=axs[d, 3])

        plt.show()
        plt.pause(0.001)

        if time_point < end - 1:
            plt.clf()


#%%
def plot_value(
    valuefun_vec,
    d,
    smin=0.1,
    smax=10,
    x=0,
    resolution=10,
    azim=230,
    elev=10,
    save=False,
    name="",
    col_max=1,
):
    """Plot the value function.

    Parameters
    ----------
    valuefun_vec : callable
        vectorized value function
    d : int
        dimension of the market
    smin : float, optional
        minimal asseet value to plot
    smax : float, optional
        maximal asseet value to plot
    x : float, optional
        x value to plot
    resolution : int, optional
        resolution of the plot
    azim : int, optional
        azimuth of the plot
    elev : int, optional
        elevation of the plot
    save : bool, optional
        save the plot
    name : str, optional
        name of the plot
    col_max : float, optional
        maximal value of the colorbar

    Returns
    -------
    None.

    """
    s = np.linspace(smin, smax, resolution)
    if d == 1:
        x = np.linspace(-1, 1, resolution)
        xgrid, sgrid = np.meshgrid(x, s)
        Z = valuefun_vec(xgrid, sgrid)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        p = ax.plot_surface(
            xgrid,
            sgrid,
            Z,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False,
        )
        p.set_clim(0, col_max)

        ax.set_xlabel("X")
        ax.set_ylabel("S")
        ax.set_zlabel("Value")
        if azim is not None:
            ax.azim = azim
        if elev is not None:
            ax.elev = elev

        # Add a color bar which maps values to colors.
        fig.colorbar(p, shrink=0.5, aspect=5)
    elif d == 2:
        sgrid1, sgrid2 = np.meshgrid(s, s)

        Z = valuefun_vec(sgrid1, sgrid2)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        p = ax.plot_surface(
            sgrid1,
            sgrid2,
            Z,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False,
        )

        ax.set_xlabel("S1")
        ax.set_ylabel("S2")
        ax.set_zlabel(f"Value for X={x}")
        if azim is not None:
            ax.azim = azim
        if elev is not None:
            ax.elev = elev

        # Add a color bar which maps values to colors.
        fig.colorbar(p, shrink=0.5, aspect=5)
    if save:
        fig.savefig(
            name + "value.pdf",
            bbox_inches="tight",
            format="pdf",
        )


#%%


def plot_probabilities(
    d,
    timerange,
    omegas,
    y,
    strategy_process,
    poss_action,
    x=None,
    relaxed=True,
    timescale=True,
    xlabel="",
    ylabel="",
    connectpaths=True,
    save=False,
    name="",
):
    """Plot the probabilities of the strategy along certain input x,y.

    Parameters
    ----------
    d : int
        dimension of the market
    timerange : int
        time range to plot
    omegas : list
        list of omegas to plot
    y : numpy array
        array of y-values to plot
    strategy_process : numpy array
        array of the strategy process, for color coding
    poss_action : list
        list of possible actions in the environment
    x : numpy array, optional
        array of the x values to plot
    relaxed : bool, optional
        plot the relaxed strategy
    timescale : bool, optional
        plot time on x axis
    xlabel : str, optional
        label of the x axis
    ylabel : str, optional
        label of the y axis
    connectpaths : bool, optional
        connect the paths with lines
    save : bool, optional
        save the plot
    name : str, optional
        name of the plot

    Returns
    -------
    None.

    """

    if timescale:
        x_grid = np.tile(np.arange(0, timerange), len(omegas))
        xlabel = "time"
    else:
        x_grid = np.reshape(x[omegas, 0:timerange], timerange * len(omegas))
    y_grid = np.reshape(np.squeeze(y[omegas, 0:timerange]), timerange * len(omegas))

    if d == 1:
        color = np.zeros((len(omegas), timerange))
        i = 0
        fig, ax = plt.subplots(d, 1)
        if relaxed:
            ax.set_title(f"probability of q = {int(poss_action[0][0])}")
            for p in strategy_process:
                color[:, i] = p[omegas, 0]
                i += 1
        else:
            color = strategy_process[omegas, :]
            ax.set_title(r"Action $q^{\theta}$ in S")
        if connectpaths:
            ax.plot(np.squeeze(y[omegas, 0:timerange]).T, color="grey", lw=0.2)
        image = ax.scatter(
            x_grid,
            y_grid,
            c=np.reshape(color, (timerange * len(omegas))),
            s=0.4,
        )

        ax.axhline(0, color="red", lw=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        image.set_clim(-1, 1)
        fig.colorbar(image, ax=ax)
    else:
        fig, ax = plt.subplots(d, 2)
        for index in range(2 * d):
            action = 1 if index < d else -1
            asset = (index % d) + 1
            color = np.zeros((len(omegas), timerange))
            i = 0

            row = index // 2
            col = index % 2

            if relaxed:
                ax[row, col].set_title(f"{action} in $S^{asset}$ ")
                for p in strategy_process:
                    color[:, i] = p[omegas, index]
                    i += 1
            else:
                color = strategy_process[:, :, index]
                ax[row, col].set_title(f"{action} in $S^{asset}$ ")

            image = ax[row, col].scatter(
                x_grid,
                y_grid,
                c=np.reshape(color, (timerange * len(omegas))),
                s=0.4,
            )

            if col > 0:
                ax[row, col].get_yaxis().set_ticks([])
            if row < d - 1:
                ax[row, col].get_xaxis().set_ticks([])

            if timescale:
                ax[row, col].axhline(0, color="red")
            else:
                ax[row, col].plot([-0, 2], [-0, 2], "r-", linewidth=1)

        fig.supxlabel(xlabel)
        fig.supylabel(ylabel)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([1, 0.2, 0.02, 0.7])
        image.set_clim(0, 1)
        fig.colorbar(image, cax=cbar_ax)
        fig.tight_layout()

    if save:
        fig.savefig(
            name + "probs_along_pv.png",
            bbox_inches="tight",
            format="png",
        )


#%%


def custom_violinplot(data, title, save, name, colors, labels, npath2test):
    """Plot a violin plot of the data.

    Parameters
    ----------
    data : list of arrays
        The data to plot.
    title : str
        The title of the plot.
    save : bool
        Whether to save the plot.
    name : str
        The name of the plot.
    colors : list of str
        The colors of the plot.
    labels : list of str
        The labels of the plot.
    npath2test : int
        The number of paths used to create data. Needed for calculation of CIs.

    Returns
    -------
    None.
    """
    means = [np.mean(x) for x in data]
    errors = [
        t.ppf(0.975, npath2test - 1) * np.std(x) / np.sqrt(npath2test - 1) for x in data
    ]
    print(f"means are {means}\n")

    fig, ax = plt.subplots(1, 1)

    vp = ax.violinplot(
        data,
        points=100,
        widths=0.7,
        showmeans=True,
        showextrema=True,
        showmedians=True,
    )
    ax.set_title(title)

    for i, b in enumerate(vp["bodies"]):
        b.set_facecolor(colors[i])
        b.set_linewidth(1)
        b.set_alpha(0.5)

    plt.xticks(np.arange(1, len(labels) + 1), labels)
    mean_ax = fig.add_axes([0.15, 1.1, 0.7, 0.1])
    mean_ax.spines["top"].set_visible(False)
    mean_ax.spines["right"].set_visible(False)
    mean_ax.spines["left"].set_visible(False)
    mean_ax.get_yaxis().set_ticks([])
    mean_ax.scatter(means, [0 for i in means], color=colors)
    mean_ax.errorbar(
        x=means, y=[0 for i in means], xerr=errors, ecolor=colors, fmt="None"
    )
    mean_ax.set_title("Means")

    if save:
        fig.savefig(
            name+'.pdf',
            bbox_inches="tight",
            format="pdf",
        )

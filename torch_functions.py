#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import datetime
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

import util

# %%


def get_latest(
    model,
    config_no,
    algo,
    select_best_A2C=False,
    filename=None,
):
    """
    get latest model from folder

    Parameters
    ----------
    model : torch.nn.Module or list of torch.nn.Module
        model to be loaded.
    config_no : int
        config number.
    algo : str
        algorithm name.
    select_best_A2C : bool, optional
        select A2C model with highest running average reward. The default is False.
    filename : str, optional
        filename of model to be loaded. The default is None.

    Returns
    -------
    model : torch.nn.Module or list of torch.nn.Module
        loaded model.
    folderpath : str
        path to folder where model is stored.
    """
    path = os.path.join(
        os.getcwd(),
        f"config_{config_no}",
    )

    if not filename:
        dirs = os.listdir(path)
        dirs.sort()
        try:
            filename = [i for i in dirs if i.startswith(f"{algo}")]
            sf = sorted(
                filename,
                key=lambda x: datetime.datetime.strptime(
                    re.search(r"(?<=_).*", x).group(0), "%d-%m-%Y_%H:%M"
                ),
            )
            filename = sf[-1]
        except IndexError:
            print(
                f"model for desired algorithm is not available. please check folder {path}"
            )
            return (None, None)

    print(f"loading files from folder: {filename} of config number {config_no}.")
    folderpath = os.path.join(
        path,
        filename + "/",
    )
    if isinstance(model, list):
        for i, m in enumerate(model):
            m.load_state_dict(torch.load(folderpath + "model_" + str(i)))
            model[i] = m
        return (model, folderpath)
    else:
        modelname = "model" if algo == "PG" else "actor"
        if select_best_A2C and algo == "A2C":
            modelname = modelname + "_best"
        model.load_state_dict(torch.load(folderpath + modelname))

    return (model, folderpath)


#%%


def train_loop(
    dataloader, model, loss_fn, optimizer, entropy_reg=None, log_wandb=False
):
    """
    train loop

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        dataloader.
    model : torch.nn.Module
        model to be trained.
    loss_fn : torch.nn.Module
        loss function.
    optimizer : torch.optim.Optimizer
        optimizer.
    entropy_reg : float, optional
        entropy regularization parameter. The default is None.
    log_wandb : bool, optional
        log to wandb. The default is False.

    Returns
    -------
    None.

    """
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):

        pred = model(X)
        if model.add_tasks:
            total_loss, loss_preds, loss_add_tasks = loss_fn(pred, y)
        else:
            total_loss = loss_fn(pred, y)
            loss_preds = total_loss
            loss_add_tasks = None

        if entropy_reg is not None:
            asset_dim = model.dim * 2
            entropy = -torch.sum(
                pred[:, 0:asset_dim] * torch.log(pred[:, 0:asset_dim]), dim=1
            ).mean()
            total_loss -= entropy_reg * entropy

            if log_wandb:
                wandb.log(
                    {
                        "entropy": entropy,
                    }
                )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            total_loss, current = total_loss.item(), batch * len(X)

            # logging to wandb
            if log_wandb:
                wandb.log(
                    {
                        "total_loss": total_loss,
                    }
                )
            to_print = f"[{current:>5d}/{size:>5d}] total_loss: {total_loss:>7f}"
            try:
                if model.add_tasks:
                    if log_wandb:
                        wandb.log(
                            {
                                "loss_preds": loss_preds.item(),
                            }
                        )
                    to_print += f", loss_preds: {loss_preds.item():>7f}"
                    for i, loss in enumerate(loss_add_tasks):
                        if log_wandb:
                            wandb.log(
                                {
                                    f"loss_add_task_{i}": loss.item(),
                                }
                            )
                        to_print += f", loss_add_task_{i}: {loss.item():>7f}"
            except:
                pass
            print(to_print)


#%%


def torch_train(
    dataset,
    model,
    loss,
    optimizer,
    epochs,
    batch_size,
    entropy_reg=None,
    log_wandb=False,
):
    """
    train model

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        dataset.
    model : torch.nn.Module
        model to be trained.
    loss : torch.nn.Module
        loss function.
    optimizer : torch.optim.Optimizer
        optimizer.
    epochs : int
        number of epochs.
    batch_size : int
        batch size.
    entropy_reg : float, optional
        entropy regularization parameter. The default is None.
    log_wandb : bool, optional
        log to wandb. The default is False.

    Returns
    -------
    None.

    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
    )
    for e in range(epochs):
        if e % 100 == 0:
            print(f"Epoch {e+1}\n-------------------------------")
        train_loop(
            loader, model, loss, optimizer, entropy_reg=entropy_reg, log_wandb=log_wandb
        )


#%%


class Actor_Net(nn.Module):
    def __init__(
        self,
        num_state,
        num_action,
    ):
        super(Actor_Net, self).__init__()
        self.add_tasks = None
        self.model = nn.Sequential(
            nn.Linear(num_state, 128),
            nn.ReLU(),
            nn.Linear(128, num_action),
            nn.Softmax(dim=-1),
        )

        self.dim = num_action // 2

    # return a probability distribution over the action space
    def forward(self, state):
        return self.model(state)


#%%


class Critic_Net(nn.Module):
    def __init__(self, num_state, actor_base=None):
        super(Critic_Net, self).__init__()

        if actor_base:
            self.model = nn.Sequential(
                actor_base.hidden,
                nn.ReLU(),
                actor_base.hidden2,
                nn.ReLU(),
                nn.Linear(actor_base.hidden2.out_features, 1),
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(num_state, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )

    # return a single value
    def forward(self, state):
        return self.model(state)


#%%


class Multitask_Actor_Net(nn.Module):
    def __init__(
        self,
        dim,
        num_state,
        num_action,
        add_tasks=None,
        twolayered=True,
    ):
        super(Multitask_Actor_Net, self).__init__()
        self.dim = dim
        self.add_tasks = add_tasks
        self.hidden = nn.Linear(num_state, 128)
        self.hidden2 = nn.Linear(128, 128) if twolayered else None
        self.output = nn.Linear(128, num_action)
        self.losses = [torch.nn.L1Loss()]

        num_add_tasks = 0
        if add_tasks is not None:

            add_task_activation_funcs = []
            add_losses = []
            for at in add_tasks:
                if at not in ["sign", "volas"]:
                    raise NotImplementedError("additional task not implemented")
                else:
                    if at == "sign":
                        num_add_tasks += 1
                        add_task_activation_funcs += [lambda x: torch.tanh(10 * x)]
                        add_losses += [torch.nn.L1Loss()]
                    elif at == "volas":
                        num_add_tasks += self.dim
                        add_task_activation_funcs += [
                            torch.nn.functional.leaky_relu
                        ] * self.dim
                        add_losses += [torch.nn.L1Loss()] * self.dim

            self.additional_outputs = [nn.Linear(128, 1) for i in range(num_add_tasks)]
            self.add_task_activation_funcs = add_task_activation_funcs
            self.losses += add_losses
            self.num_add_tasks = num_add_tasks
        else:
            self.additional_outputs = None
            self.add_task_activation_funcs = None

        self.num_add_tasks = num_add_tasks

    # return a probability distribution over the action space
    def forward(self, state):

        x = self.hidden(state)
        x = torch.relu(x)
        if self.hidden2 is not None:
            x = self.hidden2(x)
            x = torch.relu(x)
        y1 = self.output(x)
        y1 = torch.softmax(y1, dim=-1)
        if self.add_task_activation_funcs:
            y2 = [additional_output(x) for additional_output in self.additional_outputs]
            y2 = [self.add_task_activation_funcs[i](y2[i]) for i in range(len(y2))]
            y1 = torch.concat([y1] + y2, dim=-1)
        return y1

    def loss_fn(self, y_pred, y_true):
        loss = self.losses[0](
            y_pred[:, : self.output.out_features], y_true[:, : self.output.out_features]
        )

        add_losses = [
            self.losses[i + 1](
                y_pred[:, self.output.out_features + i],
                y_true[:, self.output.out_features + i],
            )
            for i in range(self.num_add_tasks)
        ]
        return (loss + sum(add_losses), loss, add_losses)

    def loss_fn_A2C(self, y_pred, y_true):
        loss = torch.zeros(1)
        for i in range(len(self.additional_outputs)):
            loss += self.losses[i + 1](y_pred[:, i], y_true[:, i])
        return loss

    def create_add_task_targets(self, X, market_env):
        if self.add_tasks is None:
            raise ValueError("no additional tasks defined")
        else:
            add_task_targets = []
            for at in self.add_tasks:
                if at == "sign":
                    add_task_targets += [-util.custom_sign(X)]
                elif at == "volas":
                    add_task_targets += [
                        torch.sqrt(torch.diag(torch.Tensor(market_env.stock.sigma)))
                        .reshape(1, -1)
                        .repeat(X.shape[0], 1)
                    ]
            return torch.cat(add_task_targets, dim=-1)


#%%


class Multitask_A2C_Net(nn.Module):
    def __init__(
        self,
        dim,
        num_state,
        num_action,
        add_tasks=None,
    ):
        super(Multitask_A2C_Net, self).__init__()
        self.dim = dim

        self.actor = Multitask_Actor_Net(
            dim=dim,
            num_state=num_state,
            num_action=num_action,
            add_tasks=add_tasks,
        )

        self.critic = Critic_Net(num_state=num_state, actor_base=self.actor)

    # return a probability distribution and a value over the action space
    def forward(self, state):
        y_actor = self.actor(state)
        y_value = self.critic(state)
        return y_actor, y_value


class DH_Net(nn.Module):
    def __init__(
        self,
        num_state,
        asset_dim,
        twolayered=False,
    ):
        super(DH_Net, self).__init__()
        self.dim = asset_dim
        self.hidden = nn.Linear(num_state, 128)
        self.hidden2 = nn.Linear(128, 128) if twolayered else None
        self.output = nn.Linear(128, asset_dim)

    # return an action in the l1 ball
    def forward(self, state):
        x = self.hidden(state)
        x = torch.relu(x)
        if self.hidden2 is not None:
            x = self.hidden2(x)
            x = torch.relu(x)
        y1 = self.output(x)
        norm = torch.linalg.norm(y1, dim=-1, ord=1).reshape(-1, 1)
        y1 = torch.where(torch.greater(norm, torch.ones(1)), torch.div(y1, norm), y1)
        return y1

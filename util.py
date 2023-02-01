#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch

# %%
def timediff_d_h_m_s(td):
    """Returns the difference between two datetime objects in days, hours, minutes and seconds.

    Parameters
    ----------
    td : datetime.timedelta
        The difference between two datetime objects.

    Returns
    -------
    int
        The number of days.
    int
        The number of hours.
    int
        The number of minutes.
    int
        The number of seconds.
    """
    # can also handle negative datediffs
    if (td).days < 0:
        td = -td
        return (
            -(td.days),
            -int(td.seconds / 3600),
            -int(td.seconds / 60) % 60,
            -(td.seconds % 60),
        )
    return td.days, int(td.seconds / 3600), int(td.seconds / 60) % 60, td.seconds % 60


# %%
def custom_sign(x):
    """Returns the sign of x, but if x is 0, returns 1 instead of 0.

    Parameters
    ----------
    x : float or np.ndarray or torch.Tensor
        The input value.

    Returns
    -------
    float or np.ndarray or torch.Tensor
    """

    if type(x) == np.float64:
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 1
    elif type(x) == np.ndarray:
        x = np.sign(x)
    elif type(x) == torch.Tensor:
        x = torch.sign(x)
    x[x == 0] = 1
    return x

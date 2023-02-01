# Machine Learning-powered Pricing of the Multidimensional Passport Option

This is a piece of software used for performing experiments on the pricing of Passport Options via reinforcement learning. The experiments are described in [1].


## A. Requirements

* Python == 3.9.13

## B. Dependencies

Prepare your python environment (`conda`, `virtualenv`, etc.), activate your environment.

Using pip:
```bash
$ pip install -r requirements.txt
```

## C. Experiments

### How to run

Choose a configuration number X and create a folder named config_X including a config.py file. This file should include dictionaries specifying parameters as follows:

```Python
import numpy as np

CONFIG_NO = X  # configuration number
SEED = 1  # random seed for numpy, random and torch

SAVE = {"model": True,  # save model after training?
        "plots": True,  # save plots?
        "plot_path": None,  # path to folder for saving plots in case SAVE['model']==False
        }  

# market params
MARKET_PARAMS = {
    "interest": 0.002,  # interest rate
    "ntime": 10,  # number of time steps per path
    "x0": 0,  # initial portfolio value
    "d": 2,  # dimension of assets
    "s0": [1, 1],  # initial values for assets, list
    "mu": [0, 0],  # mean of BM, list
    "sig": 0.04,  # var of BM, list (or float for symmetric market)
    "cor": np.eye(2),  # correlation matrix
    "additional_states": None,  # list of strings or None, additional input to strategy networks. Supported are
    # 'ret' return of assets
    # 'log_ret' log return of assets
    # 'Ssigma' volatilities*asset values
    # 'signX' sign of portfolio value
    "additional_tasks": None,  # list of strings or None, additional tasks for strategy netwoks to learn. Supported are
    # 'sign' sign of portfolio value
    # 'volas' volatilities of assets
}

# Parameters for Algorithm 3 in [1]
A2C_PARAMS = {
    "npath": 100,  # number of paths used to estimate expected reward
    "MTnetwork": False,  # should architectures of actor and critic share weights?
}
A2C_TRAIN_PARAMS = {
    "actor_lr": 0.0001,  # learning rate for actor network
    "critic_lr": 0.0007,  #learning rate for critic network
    "tau": 1e-11,  # parameter for entropy regularization
    "max_iter": 400000,  # number of iterations in A2C algorithm
    "weight_decay": 0,  # parameter for l2 regularization in all networks
    "nsteps_newPV": None,  # periodicity for sampling new initial portfolio value
    "nsteps_newS": None,  # periodicity for sampling new initial asset values
    "weights": [0, 1],  # weights for positive part and absolute value in expected reward
}
# Parameters for Algorithm 1 in [1]
PG_PARAMS = {
    "npath": 2 ** 13,  # number of paths used to estimate expected reward
    "max_t_minus": 9,  # maximal number of time steps backward
    "weights": [0, 1],  # weights for positive part and absolute value in expected reward
    "hardmax": False,  # use hardmax activation in forward pass?
}
PG_TRAIN_PARAMS = {
    "lr": 0.001,  # learning rate for Adam optimizer in PG training
    "batch_size": 2 ** 4,  # batch size in PG training
    "epochs": 2 ** 6,  # epochs per time-step in PG training
    "l2": 1e-18,  # parameter for l2 regularization in strategy network
    "entropy_reg": 1e-11,  # parameter for entropy regularization
}
# Parameters for a basic Deep Hedging (DH) algorithm
DH_PARAMS = {
    "npath": 2 ** 13,  # number of paths used to estimate expected reward
    "weights": [0, 1],  # weights for positive part and absolute value in expected reward
}
DH_TRAIN_PARAMS = {
    "lr": 0.0001,   # learning rate for Adam optimizer in DH training
    "batch_size": 2 ** 8,  # batch size in DH training
    "epochs": 2 ** 7,  # epochs in DH training
    "l2": 1e-18,  # parameter for l2 regularization in each strategy network
    "entropy_reg": 1e-18,  # parameter for entropy regularization
}
# evaluation parameters
EVAL_PARAMS = {
    "x0": 0,  # initial portfolio value for evaluation of strategies
    "use_wandb": False,  # track metrics during training via WeightsAndBiases (account neede)
    "select_best_A2C": False,  # load A2C model that yielded highest average reward during training
    "ntime": 32,  # time steps for evaluation of strategies
    "filename_A2C": None,  # name of folder from which to load trained A2C actor and critic networks
    "filename_PG": None,  # name of folder from which to load trained PG strategy network
    "filename_DH": None,  # name of folder from which to load trained DH strategy network
}
```

#### C.1 To run a policy gradient approximation (Section 3.2, Algorithm 1 in [1])
1.  Set the desired parameters for market and the PG algorithm in the file config_X/config.py.
2.  Set the config number in the notebook main-RL.ipynb.

```python
import config_X.config as config

```
3.  Run the interactive notebook main-RL.ipynb.



#### C.2 To run an A2C approximation (Section 3.3, Algorithm 3 in [1])
1. Set the desired parameters for market and the A2C algorithm in the file config_X/config.py.
2. Set the config number in the notebook main-A2C.ipynb.

```python
import config_X.config as config

```
3.  Run the interactive notebook main-A2C.ipynb.


#### C.3 To run a Deep Hedging approximation (basic implementation of [2])
1. Set the desired parameters for market and the DH algorithm in the file config_X/config.py.
2. Set the config number in the notebook mainDeepHedge.ipynb.

```python
import config_X.config as config

```
3.  Run the interactive notebook mainDeepHedge.ipynb.


#### C.4 To run a comparison of a PG and an A2C approximation
1. Set the desired parameters for market, the PG and the A2C algorithm, and the evaluation in the file config_X/config.py. Make sure to also choose the correct file names for your PG and A2C simulations in the EVAL_PARAMS dict of config_X/config.py.
2.  Set the config number in the notebook comparison.ipynb.

```python
import config_X.config as config

```
3.  Run the interactive notebook comparison.ipynb




**_NOTE:_** To enable a comparison to the experiments presented in [1]:

1. We used the configurations given in the config.py files of folders config_1 (1-dim market), config_2 (symmetric 2d market) and config_22 (asymmetric 2d market).
2.  We conducted these experiments on
    - **system:** Linux
    - **version:** Fedora release 32 (Thirty Two)
    - **platform:** Linux-5.8.12-200.fc32.x86_64-x86_64-with-glibc2.2.5
    - **machines:** Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz processors with 4 cores and 15GB RAM
    - **python:** Python 3.8.7 [GCC 10.2.1 20201125 (Red Hat 10.2.1-9)] on linux


## E. References

[1] Machine Learning-powered Pricing of the Multidimensional Passport Option

[2] [Deep Heding](https://arxiv.org/abs/1802.03042)

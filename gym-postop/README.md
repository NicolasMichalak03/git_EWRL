# Gym-PostOp

Gym-PostOp is a Gymnasium environment which simulates patient follow-up after an operation as a stabilization problem. Companion to the paper "Times Series Meet MDPs for Patient Follow Up".
| 	<!-- --> | <!-- -->  |
|-------------------|---------------------------------------|
| Action Space      | Discrete(2)                           |
| Observation Space | spaces.Box(0, np.inf, shape=(4,))     |
| import            | from gym_postop.envs import PostOpEnv |


See also the example folder for minimal script training PPO on this environment.

## Description

Motivated by the follow-Up of patients after Surgery, in Gym-Postop we simulate a time series (using ARIMA models perturbed by random event) which represents the gap between a "healthy" baseline and the observed actual physical constant of the patient (typically their weight, glucose level...). Each patient generates a unique trajectory with unknown, trajectory-specific parameters that are given as input of the environment. Each patient comes one after the other, and is tracked for a given number of steps (parameter `max_steps`).

The environment is parametrized by both known and unknown parameters denoted `true_param`: `alpha` and `phi` that we want to learn, and `known_parameters`, `sigma` and `cost` which are both known to the learner.

## Installation

To install the environment, use `pip`:

```
pip install .
```

## Example
Here is a simple example of training PPO from stable-baselines3 on PostOp-Env

```python
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from gym_postop.envs import  PostOpEnv

seed = 42
set_random_seed(seed)
max_steps = 120

model_kwargs = {'n_steps': 2**8, 'gamma': 0.9998734165520097, 'gae_lambda': 0.9802203841791037, 'learning_rate': 1.0924495921533139e-03, 'ent_coef': 0.0039720411010317576, 'max_grad_norm': 0.4508001773046434, 'policy_kwargs': {'net_arch': [{'pi': [64], 'vf': [64]}], 'activation_fn': th.nn.modules.activation.Tanh, 'ortho_init': False}}

env = PostOpEnv(max_steps = max_steps,
            true_param={'alpha': [0,0.8], "phi":0.8},
            known_parameters= {'sigma': 0.5 , 'cost': [0,200] },
            log_file="results.csv")
model = PPO("MlpPolicy",env = env, **model_kwargs, verbose=1, seed = seed, device="cpu")
model.learn(total_timesteps=1e5)
```


## Action Space

The action is discrete, deterministic with value 0 or 1. 0 indicate doing nothing and 1 is an action that tend to bring back the patient towards the baseline

## Observation Space

The observation is a ndarray with shape (4,) which gives the gap with respect to the baseline at time $t-2$, $t-1$, $t$ and the current patient time $t_p$ (i.e. number of timesteps since the current patient arrived).

## Reward

The goal is to stabilize the patient's time series around the baseline, which means having a gap as close to $0$ as possible, while managing the cost of doing an action. The reward at time $t$ is then defined as the absolute value of the gap at time $t$ plus the cost of doing the action (0 for action 0 and `known_parameters["cost"]` for action 1).

## Episode End

Each patient is tracked for a determinist amount of time given by the parameter `max_steps`, we do not consider any early stopping of an episode.

## Arguments
The environment takes the following arguments:
```
true_param: dict, default = {'alpha': [0,0.8], "phi":0.8}
   Dictionary containing the constants of the time series process that are not known to the learner. 

init: float, default=0
   Initial gap considered. 

max_steps: int, default=120
   Number of steps in an episode.

known_parameters: dict, default={'sigma': 0.5 , 'cost': [0,200] }
   Constants known to the patient: variance of the noise and cost of doing an action.

log_file: string or None
   Name of a file in which to log the times, rewards, actions and observations.
```

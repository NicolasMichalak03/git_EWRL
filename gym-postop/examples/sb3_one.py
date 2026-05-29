import glob
import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

import pandas as pd
from gym_postop.envs import ParallelPostOpEnv, PostOpEnv
import torch as th

max_steps = 120


model_kwargs = {'n_steps': 2**8, 'gamma': 0.9998734165520097, 'gae_lambda': 0.9802203841791037, 'learning_rate': 1.0924495921533139e-03, 'ent_coef': 0.0039720411010317576, 'max_grad_norm': 0.4508001773046434, 'policy_kwargs': {'net_arch': [{'pi': [64], 'vf': [64]}], 'activation_fn': th.nn.modules.activation.Tanh, 'ortho_init': False}}

os.makedirs("results", exist_ok=True)

for seed in range(20):
    env = PostOpEnv(max_steps = max_steps,
                true_param={'alpha': [0,0.8], "phi":0.8},
                known_parameters= {'sigma': 0.5 , 'cost': [0,200] },
                log_file="results/ppo_"+str(seed)+".csv")
    set_random_seed(seed)
    model = PPO("MlpPolicy",env = env, **model_kwargs, verbose=1, seed = seed, device="cpu")
    # Do the training
    model.learn(total_timesteps=1e5)

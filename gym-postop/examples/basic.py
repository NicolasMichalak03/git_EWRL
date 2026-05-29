import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from gym_postop.envs import PostOpEnv

seed = 42
set_random_seed(seed)
max_steps = 120

model_kwargs = {
    "n_steps": 2**8,
    "gamma": 0.9998734165520097,
    "gae_lambda": 0.9802203841791037,
    "learning_rate": 1.0924495921533139e-03,
    "ent_coef": 0.0039720411010317576,
    "max_grad_norm": 0.4508001773046434,
    "policy_kwargs": {
        "net_arch": [{"pi": [64], "vf": [64]}],
        "activation_fn": th.nn.modules.activation.Tanh,
        "ortho_init": False,
    },
}

env = PostOpEnv(
    max_steps=max_steps,
    true_param={"alpha": [0, 0.8], "phi": 0.8},
    known_parameters={"sigma": 0.5, "cost": [0, 200]},
    log_file="results.csv",
)
model = PPO("MlpPolicy", env=env, **model_kwargs, seed=seed)
model.learn(total_timesteps=1e5)

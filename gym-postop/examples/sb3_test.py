import gymnasium as gym
import gym_postop
from gym_postop.envs import PostOpEnv
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecMonitor
import os

SEED = 42
calendar = [1, 10, 20, 50, 100, 150, 150,  200, 300]
known_parameters = {'sigma': 0.5 , 'cout': [0,200] }
true_parameters = {'alpha': [0,0.8], "phi":0.8}
Tmax = 600

set_random_seed(SEED)
os.makedirs("results", exist_ok=True)

def make_env(d):
    return lambda: PostOpEnv(date_enter=d, max_steps=Tmax, 
                             true_param = true_parameters, 
                             parametre_connu = known_parameters,
                             log_file = f"results/log_{d}.csv")


env_fns = [ make_env(d) for d  in calendar]
envs = DummyVecEnv(env_fns)

model = PPO("MlpPolicy", envs, verbose=1)
model.learn(total_timesteps=Tmax)


# Code for Times Series Meet MDPs for Patient Follow Up

This repository contain the code for the article "Times Series Meet MDPs for Patient Follow Up" and can be used to reproduce the experiments from Section 6 and the annex. 

## Installing

To install the necessary libraries, just use `pip install -r requirements.txt`. 
The library use jax with a CPU backend and do not need use of GPU.

The experiments were conducted with Python 3.12.3

## Code files

Description of script content:
- `gym-postop`: gymnasium code for post-op environment, compatible with `stable-baselines3`.
- `mdp_jax.py`: compute discretized optimal policy through Bellman equation.
- `politiques.py`: contain definition of our estimation schemes (Gibbs sampler, Least square estimation...).
- `env.py`: environment and parallel environments.
- `simulation.py`: runner to launch experiments.

Running `simulation.py` will generate result files that can be then used to do the plots in the article by changing the parameters of the environment in `simulation.py`.

## gym-postop

gym postop can be used independently with gymnasium-compatible agents. See `gym-postop/README.md` for further informations.

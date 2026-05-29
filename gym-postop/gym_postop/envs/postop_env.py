import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.vector.sync_vector_env import SyncVectorEnv
from gymnasium.vector.vector_env import ArrayType, AutoresetMode, VectorEnv
from pettingzoo import ParallelEnv

ECART_INITIAL=0

class PostOpEnv(gym.Env):
    metadata = {"render_mode": [None], "render_fps": None}
    """
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
    """
    def __init__(self, render_mode=None, 
                 true_param={'alpha': [0,0.8], "phi":0.8},
                 init=0,
                 max_steps=120,
                 known_parameters= {'sigma': 0.5 , 'cost': [0,200] },
                 log_file = None
                 ):
        self.true_param = true_param
        self.etat_initial = init
        self.max_steps = max_steps
        self.cost = known_parameters.get('cost')
        self.sigma = known_parameters.get('sigma') or true_param.get('sigma')
        self.alpha = known_parameters.get('alpha') or true_param.get('alpha')
        self.log_file = log_file

        # Observations 
        self.observation_space = spaces.Box(0, np.inf, shape=(4,))

        if self.log_file is not None:
            with open(self.log_file, "w") as mylogfile:
                mylogfile.write("time, reward, action, observation\n")

        # We have 2 actions
        self.action_space = spaces.Discrete(2)
        self.reset()

    def _get_obs(self):
        return np.array([
            self.etat_courant,
            self.etat_precedent,
            self.old_etat_precedent,
            self.step_count,
        ], dtype=np.float32)

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.historique_costs = []
        self.historique_etats = [self.etat_initial]
        self.historique_actions = []
        self.old_etat_precedent =  self.etat_initial
        self.etat_precedent = self.etat_initial
        self.etat_courant = self.etat_initial
        self.step_count = 0
        self.done = False

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        self.historique_actions.append(action)


        alpha = self.true_param['alpha'][action]
        #beta = self.true_param['beta'][action]  #a garder si nescessaire
        phi = self.true_param['phi']
        bruit = self.np_random.normal(0, self.sigma)
        nouvel_etat = (
                    (1-alpha) * ((1 + phi) * self.etat_courant - phi * self.etat_precedent) 
                  + alpha*(phi * self.etat_courant) + bruit
        )
        # Mise à jour des états
        self.old_etat_precedent =  self.etat_precedent
        self.etat_precedent = self.etat_courant
        self.etat_courant = nouvel_etat
        self.historique_etats.append(nouvel_etat)
        cost = abs(nouvel_etat) + self.cost[action]
        self.historique_costs.append(cost)

        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True

        terminated = self.done 

        observation = self._get_obs()
        info = self._get_info()
        if self.log_file is not None:
            with open(self.log_file, "a") as mylogfile:
                mylogfile.write(f"{self.step_count}, {-cost}, {action}, {observation}\n")

        return observation, -cost, terminated, self.done, info


# Experimental: multi-Agent API from petting zoo for parallel agents.
# Agent spawn, ref: https://pettingzoo.farama.org/content/basic_usage/?utm_source=chatgpt.com#variable-numbers-of-agents-death
class ParallelPostOpEnv(ParallelEnv):
    metadata = {
        "name": "ParallelPostOpEnv",
        "render_mode": [None],
    }

    def __init__(self, calandar=[0, 10], patient_kwargs={0:[{}],10:[{},{}]}, max_n_agents=200,  max_steps=600):
        # self.patient_kwargs[t][i] contains the parameters of the ith patient arrived at timestep t
        self.patient_kwargs = patient_kwargs 
        self.calandar =  calandar
        self.timestep = None
        self.possible_agents = []
        self.max_steps=max_steps
        self.render_mode = None
        self.reset()
            
    def reset(self, seed=None, options=None):
        self.timestep = 0
        self.agents_env = {i: PostOpEnv(**kwargs, max_steps=self.max_steps) for i, kwargs in enumerate(self.patient_kwargs[0])}
        self.agents = list(self.agents_env.keys())
        self.possible_agents = list(self.agents_env.keys())
        rng = np.random.RandomState(seed)
        observations = {
            a: self.agents_env[a].reset(rng.randint(2**32))[0]
            for a in self.agents_env
        }
        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions):
        observations = {}
        rewards = {}
        dones = {}
        terminateds = {}
        for a in self.agents:
            obs, reward, terminated, done, info = self.agents_env[a].step(actions[a])
            observations[a] = obs
            rewards[a] = reward
            dones[a] = done
            terminateds[a] = terminated
        infos = {a: {} for a in self.agents_env}
        self.timestep += 1
        num_agents = len(self.agents_env)
        if self.timestep in self.calandar:
            for i, kwargs in enumerate(self.patient_kwargs[self.timestep]):
                a = i + num_agents
                self.agents_env[a] = PostOpEnv(**kwargs, max_steps = self.max_steps - self.timestep)
                self.agents.append(a)
                terminateds[a] = False
                dones[a] = False
                rewards[a] = 0
                observations[a] = self.agents_env[a]._get_obs()


        if any(dones.values()) or all(terminateds.values()):
            self.agents = []

        return observations, rewards, dones, terminateds, infos


    def render(self):
        pass

    def observation_space(self, agent):
        return self.agents_env[int(agent)].observation_space

    def action_space(self, agent):
        return self.agents_env[int(agent)].action_space

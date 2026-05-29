from pettingzoo.test import  parallel_api_test
from gym_postop.envs import ParallelPostOpEnv
env = ParallelPostOpEnv()

parallel_api_test(env, num_cycles=1_000_000)

from gymnasium.envs.registration import register

register(
    id="gym_postop/PostOp-v0",
    entry_point="gym_postop.envs:PostOpEnv",
)

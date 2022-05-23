from gym.envs.registration import register

register(
    id='carle-v0',
    entry_point='carle_gym.envs:CarleEnv',
)

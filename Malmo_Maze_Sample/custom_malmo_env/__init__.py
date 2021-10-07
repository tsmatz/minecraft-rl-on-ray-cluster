from gym.envs.registration import register

register(
    id='MalmoMazeEnv-v0',
    entry_point='custom_malmo_env.env:MalmoMazeEnv',
)

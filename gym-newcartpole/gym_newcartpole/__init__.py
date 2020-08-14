from gym.envs.registration import register

register(
    id='newcartpole-v0',
    entry_point='gym_newcartpole.envs:CartPoleNewEnv',
)
from gym.envs.registration import register

register(
    id='spm-v0',
    entry_point='gym_spm.envs:SPMenv',
)

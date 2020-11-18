from gym.envs.registration import register

register(
    id='spm_morestates-v0',
    entry_point='gym_spm_morestates.envs:SPMenv',
)

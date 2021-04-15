from gym.envs.registration import register

register(
    id='spm_morestates_discrete_action-v0',
    entry_point='gym_spm_morestates_discrete_action.envs:SPMenv',
)

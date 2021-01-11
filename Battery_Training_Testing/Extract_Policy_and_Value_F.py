import torch
import torch.nn as nn
import gym
import numpy as np
import matplotlib.pyplot as plt

# Added this to the Time Based Simulation

from stable_baselines3 import PPO, TD3, DDPG
from stable_baselines3.common.vec_env.vec_check_nan import VecCheckNan
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.cmd_util import make_vec_env
# from stable_baselines3.common.utils import set_random_seed
# from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.ddpg.policies import MlpPolicy, CnnPolicy

from stable_baselines3.common.noise import NormalActionNoise



# Instantiate Environment
# env_id = 'gym_spm:spm-v0'
# env = gym.make('gym_spm:spm-v0')
#
#
input_path = "./Model/DDGP_1.pt"
#
policy_path = "./Model/Policy.pt"
#
# value_func_path = "./Model/Value_Func.pt"
#
# n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=.75 * np.ones(n_actions))
#
# model = DDPG(MlpPolicy, env, action_noise=action_noise, verbose=1)
#
#
# model.load(input_path)

model = DDPG.load("./Model/DDGP_1.pt")

# torch.save(model, "./Model/Full_Model.pt")

model.policy.save(policy_path)

dict_thing = model.policy.state_dict()

x = torch.randn(2)

print(x.shape)
print(x)

print(model.actor)

# print(dict_thing)
# print(model.actor)
print(model.critic)
# print(model.actor_target)
# print(model.critic_target)

# thing = torch.load("./Model/Policy.pt")
# print(thing)



# print(thing['state_dict']['actor.mu.0.weight'])






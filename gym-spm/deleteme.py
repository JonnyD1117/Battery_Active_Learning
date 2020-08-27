import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, TD3
from stable_baselines3.common.vec_env.vec_check_nan import VecCheckNan
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.cmd_util import make_vec_env
# from stable_baselines3.common.utils import set_random_seed
# from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.td3.policies import MlpPolicy

from stable_baselines3.common.noise import NormalActionNoise

if __name__ == '__main__':
    env_id = 'gym_spm:spm-v0'
    num_cpu = 4  # Number of processes to use

    env = gym.make('gym_spm:spm-v0')

    print(env.observation_space)

    obs, rewards, done, info = env.step(action=0)
    print(np.size(obs))

    obs = env.reset()
    print(np.size(obs))



















# import torch


#
# device = "cuda:0"
# print(torch.version.cuda)
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name())
# print(torch.backends.cudnn.is_available())
# print(torch.backends.cudnn.enabled)



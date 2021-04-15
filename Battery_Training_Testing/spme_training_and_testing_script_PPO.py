import gym
import numpy as np
import matplotlib.pyplot as plt


from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.vec_env.vec_check_nan import VecCheckNan
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.cmd_util import make_vec_env
# from stable_baselines3.common.utils import set_random_seed
# from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from spme_battery_gym_environment_module import SPMenv

import wandb


if __name__ == '__main__':


    # env = make_vec_env(SPMenv, n_envs=1)

    env = SPMenv()
    # HyperParameters
    lr = 3e-4

    model_name = "PPO_4.pt"
    model_path = "./Model/" + model_name

    model = PPO(MlpPolicy, env, verbose=1)

    # model.learn(total_timesteps=25000)
    #
    # model.save(model_path)
    env.log_state = False

    # model = PPO.load(model_path)


    epsi_sp_list = []
    action_list = []
    soc_list = []
    Concentration_list = []
    Concentration_list1 = []

    obs = env.reset()
    for _ in range(3600):

        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

        epsi_sp_list.append(env.epsi_sp.item(0))
        soc_list.append(env.state_of_charge)
        action_list.append(action)

        if done:
            break
           # obs = env.reset()

    plt.figure()
    plt.plot(soc_list)
    plt.show()

    plt.figure()
    plt.plot(epsi_sp_list)
    plt.title("Sensitivity Values")


    print(action_list)

    plt.figure()
    plt.plot(action_list)
    plt.title("Input Currents")
    plt.show()

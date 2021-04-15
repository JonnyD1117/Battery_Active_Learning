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
from stable_baselines3.ddpg.policies import MlpPolicy

from stable_baselines3.common.noise import NormalActionNoise

import wandb


if __name__ == '__main__':
    # Instantiate Environment
    env_id = 'gym_spm_morestates:spm_morestates-v0'
    env = gym.make('gym_spm_morestates:spm_morestates-v0')

    print(env)

    # HyperParameters
    lr = 3e-4

    model_name = "DDGP_2.pt"
    model_path = "./Model/" + model_name

    # Instantiate Model
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=.75 * np.ones(n_actions))
    model = DDPG(MlpPolicy, env, action_noise=action_noise, verbose=1, train_freq=25000, n_episodes_rollout=-1)
    # model = DDPG(MlpPolicy, env, verbose=1, train_freq=2500, n_episodes_rollout=-1)

    # wandb.watch(model)


    # Train OR Load Model
    model.learn(total_timesteps=25000)
    env.log_state = False

    model.save(model_path)

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    print("Mean Reward = ", mean_reward)


    print(env.soc_list)

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

    plt.figure()
    plt.plot(action_list)
    plt.title("Input Currents")
    plt.show()

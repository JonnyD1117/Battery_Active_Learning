import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.vec_check_nan import VecCheckNan
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy

if __name__ == '__main__':
    env_id = 'gym_spm:spm-v0'
    num_cpu = 4  # Number of processes to use

    env = gym.make('gym_spm:spm-v0')


    # # env = make_vec_env(env_id, n_envs=num_cpu, seed=0)
    # # env = VecCheckNan(env, raise_exception=True)
    # print(env.action_space.low)
    # print(env.observation_space.low)
    #
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./PPO_spm_v0_tensorboard/")
    # # model.learn(total_timesteps=25000)
    model.learn(total_timesteps=25000, tb_log_name='test_run_1',)
    # model.save('PPO_test_0')

    model.load('PPO_test_0')

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    print("Mean Reward = ", mean_reward)

    epsi_sp_list = []
    action_list = []

    obs = env.reset()
    for _ in range(3600):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)

        epsi_sp_list.append(env.epsi_sp.item(0))
        action_list.append(action)

        # env.render()
    plt.figure()
    plt.plot(epsi_sp_list)
    plt.title("Sensitivity Values")

    plt.figure()
    plt.plot(action_list)
    plt.title("Input Currents")
    plt.show()

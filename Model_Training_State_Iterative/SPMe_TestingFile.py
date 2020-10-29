import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, TD3, DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise


if __name__ == '__main__':
    # Instantiate Environment
    env_id = 'gym_spm:spm-v0'
    env = gym.make('gym_spm:spm-v0')

    # Instantiate Model
    # n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=.75 * np.ones(n_actions))
    # model = DDPG(MlpPolicy, env, action_noise=action_noise, verbose=1)
    model = PPO(MlpPolicy, env, verbose=1)


    # Train OR Load Model
    model.learn(total_timesteps=1000000)

    # model.save(model_dir_description)

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

import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, TD3, DDPG, SAC
from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.sac.policies import MlpPolicy

from stable_baselines3.common.noise import NormalActionNoise

if __name__ == '__main__':
    env_id = 'gym_spm:spm-v0'
    num_cpu = 4  # Number of processes to use

    env = gym.make('gym_spm:spm-v0')

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=.75 * np.ones(n_actions))

    # model = SAC(MlpPolicy, env, action_noise=action_noise, verbose=1)
    model = SAC(MlpPolicy, env, verbose=1)

    model.learn(total_timesteps=25000)

    # model.load('DDPG_test_2_SOC_point5_two_states')
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    
    print("Mean Reward = ", mean_reward)
    
    epsi_sp_list = []
    action_list = []
    soc_list = []
    Concentration_list = []
    Concentration_list1 = []
    
    obs = env.reset()
    for _ in range(3600):
    
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
    
        epsi_sp_list.append(env.epsi_sp.item(0))
        # Concentration_list.append(env.state_output['yp'].item())
        # Concentration_list.append(env.state_output['yn'].item())
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

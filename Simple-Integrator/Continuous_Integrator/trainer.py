import numpy as np
import matplotlib.pyplot as plt

# Added this to the Time Based Simulation

from stable_baselines3 import PPO
# from stable_baselines3.common.utils import set_random_seed
# from stable_baselines3.common.vec_env import DummyVecEnv

# from stable_baselines3.td3.policies import MlpPolicy
# from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.ppo.policies import MlpPolicy

from stable_baselines3.common.noise import NormalActionNoise
from Continuous_Integrator.integrator_env import SimpleSOC


if __name__ == '__main__':

    env = SimpleSOC()

    # Instantiate Model
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=.75 * np.ones(n_actions))
    # model = DDPG(MlpPolicy, env, action_noise=action_noise, verbose=1)

    # model = DDPG(MlpPolicy, env, verbose=1, train_freq=-1, n_episodes_rollout=1, learning_starts=10000, batch_size=100)
    model = PPO(MlpPolicy, env, verbose=1, use_sde=True)
    # model.load("./model/ddpg_simp_integrators_n_eps_rollout_1_learning_starts_10k_batch_size_100_total_timesteps_200k")
    # wandb.watch(model)

    # Train OR Load Model
    model.learn(total_timesteps=200000)

    model.save(f"./log_files/models/PPO_simp_integrators_trial2")



    soc_list = []
    action_list = []
    done = False

    obs = env.reset()
    for _ in range(3600):

        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

        soc_list.append(obs.item())
        action_list.append(action)

        if done:
            break

    plt.figure()
    plt.plot(soc_list)
    plt.title("State of Charge")

    plt.figure()
    plt.plot(action_list)
    plt.title("Input Currents")
    plt.show()










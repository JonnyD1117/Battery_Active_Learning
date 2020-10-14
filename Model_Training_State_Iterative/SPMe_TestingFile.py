import gym
import numpy as np
import matplotlib.pyplot as plt


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

from stable_baselines3.common.noise import NormalActionNoise



class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=1):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log Training Variables to TB
        self.logger.record('train/Reward', env.tb_instantaneous_reward)
        self.logger.record('train/Mean Reward', env.tb_reward_mean)

        # Log Battery Variables to TB
        self.logger.record('battery/SOC', env.tb_state_of_charge)
        self.logger.record('battery/Concentration_1', env.tb_C_se0)
        self.logger.record('battery/Concentration_2', env.tb_C_se1)
        self.logger.record('battery/Sensitivity (Epsi_sp)', env.tb_epsi_sp.item())
        self.logger.record('battery/InputCurrent (Amps)', env.tb_input_current)
        self.logger.record('battery/Terminal Voltage (Volts)', env.tb_term_volt)

        # Log Time Variables to TB
        return True


if __name__ == '__main__':
    # Instantiate Environment
    env_id = 'gym_spm:spm-v0'
    env = gym.make('gym_spm:spm-v0')

    # HyperParameters
    lr = 3e-4

    # Training  & Logging Setup
    train_model = True
    train_version = 4
    description = "DDPG_Policy_MLP"

    log_dir = "./Logs/DDPG/"
    model_dir = "./Models/DDPG/"

    details = f"Model_v{train_version}_" + description

    log_dir_description = log_dir + details
    model_dir_description = model_dir + details

    # Instantiate Model
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=.75 * np.ones(n_actions))
    # model = DDPG('MlpPolicy', env, action_noise=action_noise, verbose=1, tensorboard_log=log_dir)
    model = DDPG('MlpPolicy', env, action_noise=action_noise, verbose=1)

    # model = PPO('MlpPolicy', env, tensorboard_log=log_dir)

    # Train OR Load Model
    if train_model:
        # model.learn(total_timesteps=2500000, tb_log_name=details, callback=TensorboardCallback(env), log_interval=1,)
        model.learn(total_timesteps=200000, tb_log_name=details)

        # model.save(model_dir_description)
    else:
        model.load(model_dir_description)

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

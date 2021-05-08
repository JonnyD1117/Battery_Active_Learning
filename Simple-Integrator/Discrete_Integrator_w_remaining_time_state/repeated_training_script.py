import matplotlib.pyplot as plt
# from Discrete_Integrator_w_remaining_time_state.discrete_action_integrator_w_remaining_time_env import DiscreteSimpleSOC
# from discrete_action_integrator_w_remaining_time_env import DiscreteSimpleSOC

import gym
from gym import error, spaces, utils, logger
from torch.utils.tensorboard import SummaryWriter

import random
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy


class DiscreteSimpleSOC(gym.Env):

    def __init__(self, threshold_value, trial_num, log_state=True):

        # self.cap = 2500 # mAh
        self.cap = 25.670*3600  # Ah*3600 := Amp*seconds
        self.dt = 1  # 1 second
        self.c_rate = 1

        self.threshold_value = threshold_value

        self.training_duration = 1800
        # self.training_duration = 3600
        self.remaining_time = self.training_duration

        self.time_horizon_counter = 0
        self.global_counter = 0
        self.episode_counter = 0
        self.log_state = log_state
        self.cumulative_reward = 0

        if self.log_state is True:
            self.writer = SummaryWriter(f'./log_files/REPEAT_w_time_remaining_1T{trial_num}')
            # self.writer = SummaryWriter('./log_files/tb_log/Round1/Disc_NO_TRAINING_10')

        state_limits = np.array([np.inf, np.inf], dtype=np.float32)
        action_limits = np.array([25.67*self.c_rate], dtype=np.float32)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-state_limits, state_limits, dtype=np.float32)

        self.action_dict = {0:-1, 1:0, 2: 1}
        self.state = None
        self.SOC_init = .5
        self.SOC_0 = self.SOC_init
        self.SOC = 0

        # TensorBoard Logging
        self.tb_input_current = None
        self.tb_state_of_charge = self.SOC_0
        self.tb_reward_list = []
        self.tb_reward_mean = None
        self.tb_reward_sum = None
        self.tb_instantaneous_reward = None

    def step(self, action):
        input_current = 25.67*self.action_dict[action.item()]
        # input_current = 25.67*self.action_dict[action]

        # SOC Integrator Dynamics (NOTE: Negative Input Current -> Charging)
        self.SOC = self.SOC_0 - (1./self.cap)*input_current*self.dt
        self.SOC_0 = self.SOC

        self.remaining_time += self.dt
        # Max and Min SOC Constraints
        ##############################################
        ## Testing Penalty Based SOC limiting
        ##############################################
        # if self.SOC > 1.0:
        #     self.SOC = 1.
        #
        # elif self.SOC < 0.0:
        #     self.SOC = 0.
        #
        # else:
        #     self.SOC = self.SOC
        ###############################################
        # Determine if the Episode has Reached it's termination Time
        done = bool(self.time_horizon_counter >= self.training_duration)

        if not done:
            reward = self.reward_function(self.SOC)

        else:

            logger.warn(
                  "You are calling 'step()' even though this "
                  "environment has already returned done = True. You "
                  "should always call 'reset()' once you receive 'done = "
                  "True' -- any further steps are undefined behavior.")

            reward = 0.0

        self.cumulative_reward += reward
        # Log TensorBoard Variables
        self.tb_input_current = input_current
        self.tb_state_of_charge = self.SOC
        self.tb_reward_list.append(reward)
        self.tb_reward_mean = np.mean(self.tb_reward_list)
        self.tb_reward_sum = np.sum(self.tb_reward_list)

        self.tb_instantaneous_reward = reward

        if self.log_state is True:
            self.writer.add_scalar('SimpleSOC/SOC', self.tb_state_of_charge, self.global_counter)
            self.writer.add_scalar('SimpleSOC/Input_Current', self.tb_input_current, self.global_counter)
            self.writer.add_scalar('SimpleSOC/Instant Reward', self.tb_instantaneous_reward, self.global_counter)
            self.writer.add_scalar('SimpleSOC/Cum. Reward', self.tb_reward_sum, self.global_counter)
            self.writer.add_scalar('SimpleSOC/Avg. Reward', self.tb_reward_mean, self.global_counter)
            self.writer.add_scalar('SimpleSOC/Num. Episodes', self.episode_counter, self.global_counter)

        self.state = [self.SOC, self.remaining_time]
        self.time_horizon_counter += 1
        self.global_counter += 1

        return np.array(self.state), reward, done, {"total_reward": self.cumulative_reward}

    def reset(self):

        self.episode_counter += 1
        self.time_horizon_counter = 0
        self.SOC = 0.
        self.SOC_0 = self.SOC_init
        # self.SOC_0 = random.uniform(1.2,-.2)
        # self.SOC_0 = random.uniform(1,0)
        self.cumulative_reward = 0

        self.remaining_time = 0

        # self.state = np.array([random.uniform(1,0)])

        self.state = [self.SOC_0, self.remaining_time]

        return np.array(self.state)

    def get_cumulative_reward(self):

        return self.cumulative_reward

    def reward_function(self, reward_input):

        min_reward_thres = self.threshold_value
        # min_reward_thres = .55
        max_reward_thres = .85

        # penalty =0
        #
        # reward1 = 1
        if min_reward_thres <= reward_input <= max_reward_thres:

            reward1 = 1

        else:
            reward1 = -1

        # if reward_input > 1 or reward_input < 0:
        #     penalty = -5
        #
        # reward = reward1 + penalty

        return reward1


if __name__ == '__main__':

    save_list = ["1_1", "1_2","1_3","1_4","1_5","2_1","2_2","2_3","2_4","2_5", "3_1","3_2","3_3","3_4","3_5" ]

    for ind, trial_num in enumerate(save_list):

        if ind<=4:
            thres = .55
        elif ind > 4 and ind <=9:
            thres = .6
        else:
            thres = .65

        env = DiscreteSimpleSOC(threshold_value=thres, trial_num=trial_num)
        model = DQN(MlpPolicy, env, verbose=1)
        model.learn(total_timesteps=750000)

        model.save(f"./model/REPEAT_w_time_remaining_1T{trial_num}")

        action_value = {0:-25.67, 1:0, 2: 25.67}

        soc_list = []
        remaining_time_list = []
        action_list = []
        done = False

        obs = env.reset()
        stoichastic=[True, False]

        for stoich_val in stoichastic:

            for _ in range(3600):

                action, _states = model.predict(obs, deterministic=stoich_val)

                # print(f"action is {action}")
                # print(f"_state is {_states}")

                obs, rewards, done, info = env.step(action)

                # print(f"SOC STATE: {obs[0]} ")
                # print(f"REMAINING STATE: {obs[1]} ")

                aval = action_value[action.item()]

                soc_list.append(obs[0])
                remaining_time_list.append(obs[1])
                action_list.append(aval)

                if done:
                    break

            plt.figure()
            plt.plot(soc_list)
            plt.title(f"State of Charge: Trial Number: {trial_num} Stoichastic: {stoich_val}")
            plt.savefig(f"./model/images/REPEAT_w_time_remaining_1T{trial_num}_stoich_{stoich_val}_SOC.png")
            np.save(f"./model/outputs/REPEAT_w_time_remaining_1T{trial_num}_stoich_{stoich_val}_SOC", soc_list)

            plt.figure()
            plt.plot(remaining_time_list)
            plt.title(f"Remaining Time: Trial Number: {trial_num} Stoichastic: {stoich_val}")
            plt.savefig(f"./model/images/REPEAT_w_time_remaining_1T{trial_num}_stoich_{stoich_val}_remaining_time.png")
            np.save(f"./model/outputs/REPEAT_w_time_remaining_1T{trial_num}_stoich_{stoich_val}_SOC", remaining_time_list)

            plt.figure()
            plt.plot(action_list)
            plt.title(f"Input Currents: Trial Number: {trial_num} Stoichastic: {stoich_val}")
            plt.savefig(f"./model/images/REPEAT_w_time_remaining_1T{trial_num}_stoich_{stoich_val}_input_current.png")
            np.save(f"./model/outputs/REPEAT_w_time_remaining_1T{trial_num}_stoich_{stoich_val}_SOC", action_list)

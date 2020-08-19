from SPMe_w_Sensitivity_Params import SingleParticleModelElectrolyte_w_Sensitivity
import gym
from gym import error, spaces, utils, logger
from gym.utils import seeding
import numpy as np


class SPMenv(gym.Env, SingleParticleModelElectrolyte_w_Sensitivity):

    metadata = {'render.modes': ['human']}

    def __init__(self, time_step=1, SOC=1):
        super(SingleParticleModelElectrolyte_w_Sensitivity).__init__()

        self.time_step = time_step
        self.SPMe = SingleParticleModelElectrolyte_w_Sensitivity(timestep=self.time_step)

        # state_limits = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        state_limits = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32)


        max_C_val = np.array([25.67*3], dtype=np.float32)

        self.SOC_0 = SOC
        self.state_of_charge = SOC
        self.epsi_sp = None
        self.term_volt = None

        self.min_soc = -.00000001
        self.max_soc = 1.0000001
        self.min_voltage = 2.74
        self.max_voltage = 4.1

        self.action_space = spaces.Box(-max_C_val, max_C_val, dtype=np.float32)
        self.observation_space = spaces.Box(-state_limits, state_limits, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.sim_state = None
        self.steps_beyond_done = None
        self.np_random = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def unpack_states(self, bat_states, sen_states):

        x1 = bat_states['xn']
        x2 = bat_states['xp']
        x3 = bat_states['xe']

        x4 = sen_states['Sepsi_p']
        x5 = sen_states['Sepsi_n']
        x6 = sen_states['Sdsp_p']
        x7 = sen_states['Sdsn_n']

        # return [x1[0], x1[1], x1[2], x2[0], x2[1], x2[2], x3[0], x3[1], x4[0], x4[1], x4[2], x5[0], x5[1], x5[2], x6[0], x6[1], x6[2], x6[3], x7[0], x7[1], x7[2], x7[3]]
        return [x1[0][0], x1[1][0], x1[2][0], x2[0][0], x2[1][0], x2[2][0], x3[0][0], x3[1][0], x4[0][0], x4[1][0], x4[2][0], x5[0][0], x5[1][0], x5[2][0], x6[0][0], x6[1][0], x6[2][0], x6[3][0], x7[0][0], x7[1][0], x7[2][0], x7[3][0]]

    def reward_function(self, sensitivity_value):

        reward = sensitivity_value ** 2

        return reward

    def step(self, action):
        # err_msg = "%r (%s) invalid" % (action, type(action))
        # assert self.action_space.contains(action), err_msg

        # self.SPMe.step()

        [bat_states, new_sen_states, outputs, sensitivity_outputs, soc_new, V_term, theta, docv_dCse] = self.SPMe.step(full_sim=True, states=self.sim_state, I_input=action, state_of_charge=self.state_of_charge)

        self.sim_state = [bat_states, new_sen_states]

        self.state = (self.unpack_states(bat_states, new_sen_states))
        self.epsi_sp = sensitivity_outputs['dV_dEpsi_sp']

        self.state_of_charge = soc_new[0]
        # self.state = [bat_states, new_sen_states]

        done = bool(self.state_of_charge < self.min_soc
                    or self.state_of_charge > self.max_soc
                    or V_term < self.min_voltage
                    or V_term > self.max_voltage)

        if not done:
            reward = self.reward_function(self.epsi_sp)
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = self.reward_function(self.epsi_sp)

        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                  "You are calling 'step()' even though this "
                  "environment has already returned done = True. You "
                  "should always call 'reset()' once you receive 'done = "
                  "True' -- any further steps are undefined behavior.")

            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):

        self.state = None
        self.sim_state = None
        # self.state_of_charge = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.state_of_charge = self.SOC_0

        [bat_states, new_sen_states, outputs, sensitivity_outputs, soc_new, V_term, theta, docv_dCse] = self.SPMe.step(
            full_sim=True, states=self.state, I_input=0, state_of_charge=self.state_of_charge)

        self.sim_state = [bat_states, new_sen_states]
        self.state = self.unpack_states(bat_states, new_sen_states)


        self.steps_beyond_done = None
        # return np.array(self.state)
        return self.state


   # def render(self, mode='human'):
   #
   # def close(self):


from SPMe_w_Sensitivity_Params import SingleParticleModelElectrolyte_w_Sensitivity
import gym
from gym import error, spaces, utils, logger
from gym.utils import seeding
import numpy as np

SPMe = SingleParticleModelElectrolyte_w_Sensitivity()


class SPMenv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self):

        state_limits = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32)

        max_C_val = 25.67*3

        self.state_of_charge = .5
        self.term_volt = None

        self.min_soc = 0
        self.max_soc = 1
        self.min_voltage = 2.75
        self.max_voltage = 4.2

        self.action_space = spaces.Box(-max_C_val, max_C_val, dtype=np.float32)
        self.observation_space = spaces.Box(-state_limits, state_limits, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.np_random = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reward_function(self):
        pass

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        [bat_states, new_sen_states, outputs, sensitivity_outputs, soc_new, V_term, theta, docv_dCse] = SPMe.step(full_sim=True, states=self.state, I_input=action, state_of_charge=self.state_of_charge)

        self.state = [bat_states, new_sen_states]

        done = bool(self.state_of_charge < self.min_soc
                    or self.state_of_charge > self.max_soc
                    or V_term < self.min_voltage
                    or V_term > self.max_voltage)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0

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
        # self.state_of_charge = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.state_of_charge = .5

        self.steps_beyond_done = None
        # return np.array(self.state)
        return self.state


   # def render(self, mode='human'):
   #
   # def close(self):


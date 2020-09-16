from SPMe_w_Sensitivity_Params import SingleParticleModelElectrolyte_w_Sensitivity
import gym
from gym import error, spaces, utils, logger
from gym.utils import seeding
import numpy as np


class SPMenv(gym.Env, SingleParticleModelElectrolyte_w_Sensitivity):

    # metadata = {'render.modes': ['human']}

    def __init__(self, time_step=1, SOC=.5):
        super(SingleParticleModelElectrolyte_w_Sensitivity).__init__()

        # print("INIT CALLED")

        self.max_sen = 0
        self.C_se = None

        self.time_step = time_step
        self.step_counter = 0
        self.SPMe = SingleParticleModelElectrolyte_w_Sensitivity(timestep=self.time_step)

        state_limits = np.array([np.inf, np.inf], dtype=np.float32)
        max_C_val = np.array([25.67*5], dtype=np.float32)

        self.SOC_0 = SOC
        self.state_of_charge = SOC
        self.epsi_sp = None
        self.term_volt = None

        self.min_soc = .04
        self.max_soc = 1.
        self.min_voltage = 2.74
        self.max_voltage = 4.1

        self.action_space = spaces.Box(-max_C_val, max_C_val, dtype=np.float32)
        self.observation_space = spaces.Box(-state_limits, state_limits, dtype=np.float32)
        # self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,) , dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.sim_state = None
        self.steps_beyond_done = None
        self.np_random = None
        self.state_output = None
        self.re = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @staticmethod
    def unpack_states(bat_states, sen_states, state_out, sen_out):

        x1 = bat_states['xn']
        x2 = bat_states['xp']
        x3 = bat_states['xe']

        x4 = sen_states['Sepsi_p']
        x5 = sen_states['Sepsi_n']
        x6 = sen_states['Sdsp_p']
        x7 = sen_states['Sdsn_n']

        yn = state_out["yn"]
        yp = state_out["yp"]
        yep = state_out["yep"]

        dV_dDsn = sen_out["dV_dDsn"]
        dV_dDsp = sen_out["dV_dDsp"]
        dCse_dDsn = sen_out["dCse_dDsn"]
        dCse_dDsp = sen_out["dCse_dDsp"]
        dV_dEpsi_sn = sen_out["dV_dEpsi_sn"]
        dV_dEpsi_sp = sen_out["dV_dEpsi_sp"]

        return [yp.item(), dV_dEpsi_sp.item()]

    def reward_function(self, sensitivity_value, action):

        reward = sensitivity_value**2
        # if action >= 25.67*3 or action <= -25.67*3:
        #     action_penalty = -100
        # else:
        #     action_penalty = 0
        #
        # if np.abs(sensitivity_value) > self.max_sen:
        #
        #     sen_reward = 1
        #     self.max_sen = sensitivity_value
        # else:
        #     sen_reward = 0
        #
        # reward = action_penalty + sen_reward

        return reward

    def step(self, action):
        # err_msg = "%r (%s) invalid" % (action, type(action))
        # assert self.action_space.contains(action), err_msg

        if self.step_counter == 0:
            self.sim_state = self.SPMe.full_init_state
            self.step_counter += 1
        else: 
            self.step_counter += 1

        # [bat_states, new_sen_states, outputs, sensitivity_outputs, soc_new, V_term, theta, docv_dCse, done_flag] \
        #     = self.SPMe.step(full_sim=True, states=self.sim_state, I_input=action, state_of_charge=self.state_of_charge)

        [bat_states, new_sen_states, outputs, sensitivity_outputs, soc_new, V_term, theta, docv_dCse, done_flag] \
            = self.SPMe.step(full_sim=True, states=self.sim_state, I_input=action)

        self.sim_state = [bat_states, new_sen_states]
        self.state_output = outputs
        self.state = self.unpack_states(bat_states, new_sen_states, outputs, sensitivity_outputs)
        self.epsi_sp = sensitivity_outputs['dV_dEpsi_sp']
        self.term_volt = V_term

        # print(self.epsi_sp.item())

        self.C_se = theta[1].item()
        self.state_of_charge = soc_new[1].item()
        # self.state = [bat_states, new_sen_states]

        done = bool(self.state_of_charge < self.min_soc
                    or self.state_of_charge > self.max_soc
                    or np.isnan(V_term)
                    or done_flag is True)

        # if done is True:
        #     pass
        #     # # print("GYM env 'STEP' returned DONE = TRUE. Exit current Simulation & Reset")
        #     # self.reset()

        if not done:
            reward = self.reward_function(self.epsi_sp.item(), action)
            self.re = reward

        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = self.reward_function(self.epsi_sp.item(), action)
            self.re = reward

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
        self.step_counter = 0
        # print("RESET CALLED")
        self.state = None

        # self.state_of_charge = np.random.uniform(low=.25, high=.98)
        self.state_of_charge = self.SOC_0
        self.SPMe.__init__(init_soc=self.state_of_charge)

        self.sim_state = self.SPMe.full_init_state

        # print(self.sim_state)

        [bat_states, new_sen_states, outputs, sensitivity_outputs, soc_new, V_term, theta, docv_dCse, done] = self.SPMe.step(
            full_sim=True, states=self.sim_state, I_input=0)

        self.sim_state = [bat_states, new_sen_states]
        self.state = self.unpack_states(bat_states, new_sen_states, outputs, sensitivity_outputs)

        self.steps_beyond_done = None
        return np.array(self.state)


   # def render(self, mode='human'):
   #
   # def close(self):

if __name__ == '__main__':

    gym = SPMenv()

    gym.reset()

import gym
from matplotlib import pyplot as plt
import scipy.io
import numpy as np

# Initialize Custom Gym Representation from OpenAI GYM Registration
custom_gym = gym.make('gym_spm:spm-v0')
custom_gym.__init__(time_step=.2, SOC=.7)       # Re-initialize to recompute with "custom" parameters (time_step, init_soc)

# Declare Empty lists to store simulation results.
SOC_list = []
epsi_sp_list = []
reward_list = []

time = np.arange(0, 2000.2, .2)
I_sine = [100*np.sin(.001*t) for t in range(len(time))]

plt.plot(I_sine)
plt.show()

# Iterate over Custom Battery Environment for duration of I_FUDS profile
for t in range(0, len(time)-1):

    # Use OpenAI GYM "STEP" func. to propagate inputs through battery model
    states, reward, done, info = custom_gym.step(I_sine[t])

    # STEP func.
    if done:
        break

    SOC_list.append(custom_gym.state_of_charge[0].item())
    epsi_sp_list.append(custom_gym.epsi_sp.item(0))
    reward_list.append(reward.item())

plt.figure()
plt.xlabel('Time [seconds]')
plt.ylabel('State of Charge (SOC)')
plt.title('SOC vs Time [FUDs Profile: SOC_0 = .7] ')
plt.plot( SOC_list)

plt.figure()
plt.xlabel('Time [seconds]')
plt.ylabel('Epsilon_SP')
plt.title('Epsilon_SP vs Time [FUDs Profile: SOC_0 = .7] ')
plt.plot( epsi_sp_list)

# plt.figure()
# plt.plot(reward_list)
plt.show()



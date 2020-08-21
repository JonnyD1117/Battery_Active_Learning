import gym
from matplotlib import pyplot as plt
import scipy.io

# Initialize Custom Gym Representation from OpenAI GYM Registration
custom_gym = gym.make('gym_spm:spm-v0')
custom_gym.__init__(time_step=1, SOC=1)       # Re-initialize to recompute with "custom" parameters (time_step, init_soc)

# Declare Empty lists to store simulation results.
SOC_list_1C = []
epsi_sp_list_1C = []

SOC_list_2C = []
epsi_sp_list_2C = []

SOC_list_3C = []
epsi_sp_list_3C = []

# time = 3600//2
# I = 25.67*2

e_state = []
e_output = []

I_list = [25.67*1, 25.67*2, 25.67*3]
time_list = [3600, 1800, 1200]

I_list = [25.67*1]



for ind in range(len(I_list)):

    time = time_list[ind]
    I = I_list[ind]

    # Iterate over Custom Battery Environment for duration of Constant Current profile
    for t in range(0, time):

        # Use OpenAI GYM "STEP" func. to propagate inputs through battery model
        states, reward, done, info = custom_gym.step(I)

        e_state.append(states[6])

        # STEP func.
        if done:
            break

        if ind == 0:
            SOC_list_1C.append(custom_gym.state_of_charge[0].item())
            epsi_sp_list_1C.append(custom_gym.epsi_sp.item(0) * custom_gym.SPMe.param['epsilon_sp'])

        if ind == 1:
            SOC_list_2C.append(custom_gym.state_of_charge[0].item())
            epsi_sp_list_2C.append(custom_gym.epsi_sp.item(0) * custom_gym.SPMe.param['epsilon_sp'])

        if ind == 2:
            SOC_list_3C.append(custom_gym.state_of_charge[0].item())
            epsi_sp_list_3C.append(custom_gym.epsi_sp.item(0) * custom_gym.SPMe.param['epsilon_sp'])
    custom_gym.reset()
    # break


plt.figure()
plt.plot(e_state)
plt.show()
print(SOC_list_1C)
plt.figure()
plt.xlabel('Time [seconds]')
plt.ylabel('State of Charge (SOC)')
plt.title('SOC vs Time [CC Profile: SOC_0 = 1] ')
plt.plot(SOC_list_1C, "-b", label="SOC @ 1C")
plt.plot(SOC_list_2C, "-r", label="SOC @ 2C")
plt.plot(SOC_list_3C, label="SOC @ 3C")
plt.legend()

plt.figure()
plt.xlabel('Time [seconds]')
plt.ylabel('Epsilon_SP')
plt.title('Epsilon_SP vs Time [CC Profile: SOC_0 = 1] ')
plt.plot(epsi_sp_list_1C, "-b", label="Sen @ 1C")
plt.plot(epsi_sp_list_2C, "-r", label="Sen @ 2C")
plt.plot(epsi_sp_list_3C,  label="Sen @ 3C")
plt.legend()
plt.show()


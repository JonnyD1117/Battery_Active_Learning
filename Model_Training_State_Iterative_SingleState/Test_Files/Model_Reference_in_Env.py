import scipy.io
from SPMe_w_Sensitivity_Params import SingleParticleModelElectrolyte_w_Sensitivity
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import gym

import numpy as np


if __name__ == '__main__':

    # Created Environment from Register
    env_id = 'gym_spm:spm-v0'
    env = gym.make('gym_spm:spm-v0')

    # Import FUDs profile
    mat = scipy.io.loadmat("I_FUDS.mat")
    I_fuds = mat["I"][0][:]

    # Import Pulse Profile
    mat2 = scipy.io.loadmat("Test_Data_mfiles/correct_pulse_input.mat")
    I_pulse = mat2["I"][0][:]

    I_fuds = I_pulse


    # plt.plot(I_fuds)
    # plt.show()

    # obs = env.reset()
    counter = 0
    for action in I_fuds:

        action = np.array(25.67*3)
        obs, rewards, done, info = env.step(action)


        counter += 1

        if done:
            break

    mfile_data = {'input_current': env.rec_input_current, 'SOC': env.rec_state_of_charge, 'time': env.rec_time, 'V_term': env.rec_term_volt, 'Epsi_sp': env.rec_epsi_sp}

    # scipy.io.savemat("Env_3C_CC_Mat_File.mat", mfile_data)

    plt.figure(1)
    plt.plot(env.rec_state_of_charge)
    plt.title("3C CC Discharge: SOC")
    plt.xlabel('Num Time Steps')
    plt.ylabel('SOC')

    plt.figure(2)
    plt.plot(env.rec_input_current)
    plt.title("3C CC Discharge: Current")
    plt.xlabel('Num Time Steps')
    plt.ylabel('Current [amps]')

    plt.figure(3)
    plt.plot(env.rec_term_volt)
    plt.title("3C CC Discharge: Terminal Voltage")
    plt.xlabel('Num Time Steps')
    plt.ylabel('V_term [volts]')

    plt.figure(4)
    plt.plot(env.rec_epsi_sp)
    plt.title("3C CC Discharge: dV_dEpsi_sp")
    plt.xlabel('Num Time Steps')
    plt.ylabel('Sensitivity')
    plt.show()

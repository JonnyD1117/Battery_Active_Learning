from SPMe_w_Sensitivity_Params import SingleParticleModelElectrolyte_w_Sensitivity
import scipy.io
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    mat = scipy.io.loadmat("I_FUDS.mat")
    I_fuds = mat["I"][0][:]


    SPMe = SingleParticleModelElectrolyte_w_Sensitivity()
    thing = np.inf
    states = None
    I = -25.67*1
    time = 3600

    SOC_0 = 0
    term_voltage = np.zeros(time,)
    rec_states = np.zeros(time,)
    rec_soc = np.zeros(time, )
    rec_docv_dCse = np.zeros(time,)

    bat_states = {"xn": None, "xp": None, "xe": None}
    sensitivity_states = {"Sepsi_p": None, "Sepsi_n": None, "Sdsp_p": None, "Sdsn_n": None}

    for t in range(0, time):

        [bat_states, new_sen_states, outputs, sensitivity_outputs, soc_new, V_term, theta, docv_dCse] = SPMe.step(full_sim=True, states=states, I_input=I, state_of_charge=SOC_0)

        states = [bat_states, new_sen_states]
        SOC_0 = soc_new

        term_voltage[t] = V_term
        rec_soc[t] = soc_new[0]
        rec_docv_dCse[t] = docv_dCse[0]
        # rec_states[t] = states


    plt.figure()
    plt.plot(term_voltage)
    plt.figure()
    plt.plot(rec_soc)
    plt.figure()
    plt.plot(rec_docv_dCse)
    plt.show()











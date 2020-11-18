from SPMe_w_Sensitivity_Params import SingleParticleModelElectrolyte_w_Sensitivity
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import scipy.io
import numpy as np


SPMe_Model = SingleParticleModelElectrolyte_w_Sensitivity(timestep=1,init_soc=1)

# Import FUDs profile
mat = scipy.io.loadmat("I_FUDS.mat")
I_fuds = mat["I"][0][:]
time_fuds = mat["time"][0][:]

mat2 = scipy.io.loadmat("Test_Data_mfiles/correct_pulse_input.mat")
I_pulse = mat2["I"][0][:]

print(len(time_fuds))

dt_f = .2
k = 10000

total_time = k*dt_f

zero_input = np.zeros(10000)


[xn, xp, xe, yn, yp, yep, theta_n, theta_p, docv_dCse_n, docv_dCse_p, V_term,
time, current, soc, dV_dDsn, dV_dDsp, dCse_dDsn, dCse_dDsp, dV_dEpsi_sn, dV_dEpsi_sp]\
= SPMe_Model.sim(CC=True, zero_init_I=False, I_input=[25.67*3], init_SOC=1, trim_results=False, plot_results=False,sim_time=1100, delta_t=1)


mfile_data = {'input_current':current, 'SOC': soc, 'time':time, 'V_term':V_term, 'Epsi_sp':dV_dEpsi_sp}

scipy.io.savemat("Test_Data_mfiles/Direct_3C_CC_Mat_File", mfile_data, appendmat=True)

plt.figure(1)
plt.plot(soc)
plt.title("2C CC Discharge: SOC")
plt.xlabel('Num Time Steps')
plt.ylabel('SOC')

plt.figure(2)
plt.plot(current)
plt.title("2C CC Discharge: Current")
plt.xlabel('Num Time Steps')
plt.ylabel('Current [amps]')

plt.figure(3)
plt.plot(V_term)
plt.title("2C CC Discharge: Terminal Voltage")
plt.xlabel('Num Time Steps')
plt.ylabel('V_term [volts]')

plt.figure(4)
plt.plot(dV_dEpsi_sp)
plt.title("2C CC Discharge: dV_dEpsi_sp")
plt.xlabel('Num Time Steps')
plt.ylabel('Sensitivity')
plt.show()

import scipy.io
# from SPMe_w_Sensitivity_Params import SingleParticleModelElectrolyte_w_Sensitivity
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


mat = scipy.io.loadmat("I_FUDS.mat")
mat2 = scipy.io.loadmat("Test_Data_mfiles/correct_pulse_input.mat")

print(mat)

I_fuds = mat["I"][0][:]
time_fuds = mat['time'][0][:]

I_pulse = mat2["I"][0][:]
time_pulse = mat2['time'][0][:]

dt_f = time_fuds[20] - time_fuds[19]
dt_p = time_pulse[20] - time_pulse[19]

print(f"FUDs Time Step: {time_fuds[20] - time_fuds[19]}")
print(f"Pulses Time Step: {time_pulse[20] - time_pulse[19]}")

plt.figure(1)
plt.plot(time_fuds, I_fuds)
plt.title(f"Reference FUDs Input Current (TimeStep: .2 sec)")
plt.xlabel("Time [sec]")
plt.ylabel("Input Current [amps]")

plt.figure(2)
plt.plot(time_pulse, I_pulse)
plt.title(f"Reference Pulsed Input Current Signal (TimeStep: .1 sec)")
plt.xlabel("Time [sec]")
plt.ylabel("Input Current [amps]")
plt.show()


# SPMe_FUDS = SingleParticleModelElectrolyte_w_Sensitivity(sim_time=len(time_fuds)*.2, timestep=.2)
# SPMe_Pulses = SingleParticleModelElectrolyte_w_Sensitivity(sim_time=len(time_pulse), timestep=1)

# [xn, xp, xe, yn, yp, yep, theta_n, theta_p, docv_dCse_n, docv_dCse_p, V_term,
# time, current, soc, dV_dDsn, dV_dDsp, dCse_dDsn, dCse_dDsp, dV_dEpsi_sn, dV_dEpsi_sp]\
# = SPMe_Pulses.sim(CC=False, zero_init_I=False, I_input=I_pulse, init_SOC=.5, trim_results=True, plot_results=True)
#
# [xn, xp, xe, yn, yp, yep, theta_n, theta_p, docv_dCse_n, docv_dCse_p, V_term,
# time, current, soc, dV_dDsn, dV_dDsp, dCse_dDsn, dCse_dDsp, dV_dEpsi_sn, dV_dEpsi_sp]\
# = SPMe_FUDS.sim(CC=False, zero_init_I=False, I_input=I_pulse, init_SOC=.5, trim_results=True, plot_results=True)

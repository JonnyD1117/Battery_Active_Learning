
""" # mat = scipy.io.loadmat("I_FUDS.mat")
 # mat2 = scipy.io.loadmat("correct_pulse_input.mat")
 #
 # I_fuds = mat["I"][0][:]
 # # I_fuds = I_fuds[0][:]
 #
 # time_fuds = mat['time'][0][:]
 #
 # # time_fuds = time_fuds[0][:]
 #
 # I_pulse = mat2["I"][0][:]
 # # I_pulse = I_pulse[0][:]
 # time_pulse = mat2['time'][0][:]
 # # time_pulse = time_pulse[0][:]
 #
 # print(time_fuds[2]-time_fuds[1])
 # print(len(time_fuds))
 # print(time_fuds[0])
 # print(time_fuds[-1])
 #
 # input_profile = []
 #
 # # print(time_pulse[22] - time_pulse[21])
 # # print(len(time_pulse))
 # # print(time_pulse[0])
 # # print(time_pulse[-1])
 # #
 # # plt.figure(0)
 # # plt.plot(I_pulse)
 # # plt.show()
 # #
 # # print("Time Length", len(time_fuds))
 #
 #
 # # SPMe = SingleParticleModelElectrolyte_w_Sensitivity(sim_time=len(time_fuds)*.2, timestep=.2)
 # SPMe = SingleParticleModelElectrolyte_w_Sensitivity(sim_time=len(time_pulse), timestep=1)
 #
 # [xn, xp, xe, yn, yp, yep, theta_n, theta_p, docv_dCse_n, docv_dCse_p, V_term,
 # time, current, soc, dV_dDsn, dV_dDsp, dCse_dDsn, dCse_dDsp, dV_dEpsi_sn, dV_dEpsi_sp]\
 # = SPMe.sim(CC=False, zero_init_I=False, I_input=I_pulse, init_SOC=.5, trim_results=True, plot_results=True)
 #
 #
 #
 # # scipy.io.savemat("Fuds_current.mat", {"Input_current": current})"""







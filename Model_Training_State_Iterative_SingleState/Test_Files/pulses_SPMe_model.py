# SPMe_FUDS = SingleParticleModelElectrolyte_w_Sensitivity(sim_time=len(time_fuds)*.2, timestep=.2)
# SPMe_Pulses = SingleParticleModelElectrolyte_w_Sensitivity(sim_time=len(time_pulse), timestep=1)

# [xn, xp, xe, yn, yp, yep, theta_n, theta_p, docv_dCse_n, docv_dCse_p, V_term,
# time, current, soc, dV_dDsn, dV_dDsp, dCse_dDsn, dCse_dDsp, dV_dEpsi_sn, dV_dEpsi_sp]\
# = SPMe_Pulses.sim(CC=False, zero_init_I=False, I_input=I_pulse, init_SOC=.5, trim_results=True, plot_results=True)
#
# [xn, xp, xe, yn, yp, yep, theta_n, theta_p, docv_dCse_n, docv_dCse_p, V_term,
# time, current, soc, dV_dDsn, dV_dDsp, dCse_dDsn, dCse_dDsp, dV_dEpsi_sn, dV_dEpsi_sp]\
# = SPMe_FUDS.sim(CC=False, zero_init_I=False, I_input=I_pulse, init_SOC=.5, trim_results=True, plot_results=True)
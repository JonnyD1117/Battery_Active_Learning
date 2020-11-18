# from SPMeBatteryParams import *
import numpy as np
import matplotlib.pyplot as plt
from math import asinh, tanh, cosh
from SPMe_Baseline_Params import SPMe_Baseline_Parameters


class SingleParticleModelElectrolyte_w_Sensitivity(SPMe_Baseline_Parameters):
    def __init__(self, init_soc=.5, custom_params=None, timestep=1, sim_time=3600, voltage_limiter=True):

        # print("Battery model CLASS Init Called")
        self.limit_term_volt = voltage_limiter
        self.SOC_0 = init_soc

        # Initialize Default Parameters
        self.param = {}
        self.param_key_list = ['epsilon_sn', 'epsilon_sp', 'epsilon_e_n', 'epsilon_e_p',
                               'F', 'Rn', 'Rp', 'R', 'T', 'Ar_n', 'Ar_p', 'Ln', 'Lp', 'Lsep', 'Lc',
                               'Ds_n', 'Ds_p', 'De', 'De_p', 'De_n', 'kn', 'kp', 'stoi_n0', 'stoi_n100',
                               'stoi_p0', 'stoi_p100', 'SOC', 'cs_max_n', 'cs_max_p', 'Rf', 'as_n', 'as_p',
                               'Vn', 'Vp', 't_plus', 'cep', 'cen', 'rfa_n', 'rfa_p', 'epsi_sep', 'epsi_e',
                               'epsi_n', 'gamak', 'kappa', 'kappa_eff', 'kappa_eff_sep']
        self.default_param_vals = [self.epsilon_sn, self.epsilon_sp, self.epsilon_e_n, self.epsilon_e_p,
                                   self.F, self.Rn, self.Rp, self.R, self.T, self.Ar_n, self.Ar_p,  self.Ln,
                                   self.Lp, self.Lsep, self.Lc, self.Ds_n, self.Ds_p, self.De, self.De_p,
                                   self.De_n, self.kn, self.kp, self.stoi_n0, self.stoi_n100, self.stoi_p0,
                                   self.stoi_p100, self.SOC, self.cs_max_n, self.cs_max_p, self.Rf, self.as_n,
                                   self.as_p, self.Vn, self.Vp, self.t_plus, self.cep, self.cen, self.rfa_n,
                                   self.rfa_p, self.epsi_sep, self.epsi_e, self.epsi_n, self.gamak, self.kappa,
                                   self.kappa_eff, self.kappa_eff_sep]

        if custom_params is not None:
            self.param = self.import_custom_parameters(custom_params)

        else:
            self.param = {self.param_key_list[i]: self.default_param_vals[i] for i in range(0, len(self.param_key_list))}

        # Initialize the "battery" and 'sensitivity' states (FOR STEP METHOD)
        self.initial_state = self.compute_init_states()

        Sepsi_p_old = np.array([[0], [0], [0]])
        Sepsi_n_old = np.array([[0], [0], [0]])
        Sdsp_p_old = np.array([[0], [0], [0], [0]])
        Sdsn_n_old = np.array([[0], [0], [0], [0]])

        self.initial_sen_state = {"Sepsi_p": Sepsi_p_old, "Sepsi_n": Sepsi_n_old, "Sdsp_p": Sdsp_p_old,
                                  "Sdsn_n": Sdsn_n_old}

        # Initialize the "system" states for use in SIM method
        self.full_init_state = [self.initial_state, self.initial_sen_state]
        self.next_full_state = [self.initial_state, self.initial_sen_state]

        self.INIT_States = self.full_init_state

        # Simulation Settings
        self.dt = timestep
        self.simulation_time = sim_time
        self.num_steps = self.simulation_time//self.dt

        Ts = self.dt

        # Default Input "Current" Settings
        self.default_current = 25.67            # Base Current Draw

        # self.C_rate = C_Rate
        self.C_rate_list = {"1C": 3601, "2C": 1712, "3C": 1083, "Qingzhi_C": 1300}

        self.CC_input_profile = self.default_current*np.ones(self.C_rate_list["Qingzhi_C"]+1)
        self.CC_input_profile[0] = 0

        # Model Parameters & Variables
        ###################################################################
        # Positive electrode three-state state space model for the particle
        self.Ap = np.array([[0, 1, 0], [0, 0, 1], [0, -(3465 * (self.param['Ds_p'] ** 2) / self.param['Rp'] ** 4), - (189 * self.param['Ds_p'] / self.param['Rp'] ** 2)]])
        self.Bp = np.array([[0], [0], [-1]])
        self.Cp = self.param['rfa_p'] * np.array([[10395 * self.param['Ds_p'] ** 2, 1260 * self.param['Ds_p'] * self.param['Rp'] ** 2, 21 * self.param['Rp'] ** 4]])
        self.Dp = np.array([0])

        # Positive electrode SS Discretized
        [n_pos, m_pos] = np.shape(self.Ap)
        self.A_dp = np.eye(n_pos) + self.Ap * Ts
        self.B_dp = self.Bp * Ts
        self.C_dp = self.Cp
        self.D_dp = self.Dp

        # Negative electrode three-state state space model for the particle
        self.An = np.array([[0, 1, 0], [0, 0, 1], [0, - (3465 * (self.param['Ds_n'] ** 2) / self.param['Rn'] ** 4), - (189 * self.param['Ds_n'] / self.param['Rn'] ** 2)]])
        self.Bn = np.array([[0], [0], [-1]])
        self.Cn = self.param['rfa_n'] * np.array([[10395 * self.param['Ds_n'] ** 2, 1260 * self.param['Ds_n'] * self.param['Rn'] ** 2, 21 * self.param['Rn'] ** 4]])
        self.Dn = np.array([0])

        # Negative electrode SS Discretized
        [n_neg, m_neg] = np.shape(self.An)
        self.A_dn = np.eye(n_neg) + self.An * Ts
        self.B_dn = self.Bn * Ts
        self.C_dn = self.Cn
        self.D_dn = self.Dn

        # electrolyte  concentration (boundary)
        a_p0 = -(self.param['epsi_n'] ** (3 / 2) + 4 * self.param['epsi_sep'] ** (3 / 2)) / (80000 * self.param['De_p'] * self.param['epsi_n'] ** (3 / 2) * self.param['epsi_sep'] ** (3 / 2))
        b_p0 = (self.param['epsi_n'] ** 2 * self.param['epsi_sep'] + 24 * self.param['epsi_n'] ** 3 + 320 * self.param['epsi_sep'] ** 3 + 160 * self.param['epsi_n'] ** (3 / 2) * self.param['epsi_sep'] ** (3 / 2)) / (19200000000 * (4 * self.param['De_p'] * self.param['epsi_n'] ** (1 / 2) * self.param['epsi_sep'] ** 3 + self.param['De_p'] * self.param['epsi_n'] ** 2 * self.param['epsi_sep'] ** (3 / 2)))

        a_n0 = (self.param['epsi_n'] ** (3 / 2) + 4 * self.param['epsi_sep'] ** (3 / 2)) / (80000 * self.param['De'] * self.param['epsi_n'] ** (3 / 2) * self.param['epsi_sep'] ** (3 / 2))
        b_n0 = (self.param['epsi_n'] ** 2 * self.param['epsi_sep'] + 24 * self.param['epsi_n'] ** 3 + 320 * self.param['epsi_sep'] ** 3 + 160 * self.param['epsi_n'] ** (3 / 2) * self.param['epsi_sep'] ** (3 / 2)) / (19200000000 * (4 * self.param['De_n'] * self.param['epsi_n'] ** (1 / 2) * self.param['epsi_sep'] ** 3 + self.param['De_n'] * self.param['epsi_n'] ** 2 * self.param['epsi_sep'] ** (3 / 2)))

        self.Aep = np.array([[-1 / b_p0, 0], [0, -1 / b_n0]])
        self.Bep = self.param['gamak'] * np.array([[1], [1]])
        self.Cep = np.array([[a_p0 / b_p0, 0], [0, a_n0 / b_n0]])
        self.Dep = np.array([0])

        [n_elec, m] = np.shape(self.Aep)
        self.Ae_dp = np.eye(n_elec) + self.Aep * Ts
        self.Be_dp = self.Bep * Ts
        self.Ce_dp = self.Cep
        self.De_dp = self.Dep

        # Sensitivities
        # sensitivity realization in time domain for epsilon_sp from third order pade(you can refer to my slides)
        coefp = 3 / (self.param['F'] * self.param['Rp'] ** 6 * self.param['as_p'] ** 2 * self.param['Ar_p'] * self.param['Lp'])
        self.Sepsi_A_p = np.array([[0, 1, 0], [0, 0, 1], [0, -(3465 * self.param['Ds_p'] ** 2) / self.param['Rp'] ** 4, -(189 * self.param['Ds_p']) / self.param['Rp'] ** 2]])
        self.Sepsi_B_p = np.array([[0], [0], [1]])
        self.Sepsi_C_p = coefp * np.array([10395 * self.param['Ds_p'] ** 2, 1260 * self.param['Ds_p'] * self.param['Rp'] ** 2, 21 * self.param['Rp'] ** 4])
        self.Sepsi_D_p = np.array([0])

        [n, m] = np.shape(self.Sepsi_A_p)
        self.Sepsi_A_dp = np.eye(n) + self.Sepsi_A_p * Ts
        self.Sepsi_B_dp = self.Sepsi_B_p * Ts
        self.Sepsi_C_dp = self.Sepsi_C_p
        self.Sepsi_D_dp = self.Sepsi_D_p

        # sensitivity realization in time domain for epsilon_sn from third order pade(you can refer to my slides)
        coefn = 3 / (self.param['F'] * self.param['Rn'] ** 6 * self.param['as_n'] ** 2 * self.param['Ar_n'] * self.param['Ln'])

        self.Sepsi_A_n = np.array([[0, 1, 0], [0, 0, 1], [0, -(3465 * self.param['Ds_n'] ** 2) / self.param['Rn'] ** 4, -(189 * self.param['Ds_n']) / self.param['Rn'] ** 2]])
        self.Sepsi_B_n = np.array([[0], [0], [1]])
        self.Sepsi_C_n = coefn * np.array([10395 * self.param['Ds_n'] ** 2, 1260 * self.param['Ds_n'] * self.param['Rn'] ** 2, 21 * self.param['Rn'] ** 4])
        self.Sepsi_D_n = np.array([0])

        [n, m] = np.shape(self.Sepsi_A_n)
        self.Sepsi_A_dn = np.eye(n) + self.Sepsi_A_n * Ts
        self.Sepsi_B_dn = self.Sepsi_B_n * Ts
        self.Sepsi_C_dn = self.Sepsi_C_n
        self.Sepsi_D_dn = self.Sepsi_D_n

        # sensitivity realization in time domain for D_sp from third order pade
        coefDsp = (63 * self.param['Rp']) / (self.param['F'] * self.param['as_p'] * self.param['Ar_p'] * self.param['Lp'] * self.param['Rp'] ** 8)

        self.Sdsp_A = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                           [-(12006225 * self.param['Ds_p'] ** 4) / self.param['Rp'] ** 8, -1309770 * self.param['Ds_p'] ** 3 / self.param['Rp'] ** 6,
                            -42651 * self.param['Ds_p'] ** 2 / self.param['Rp'] ** 4, -378 * self.param['Ds_p'] / self.param['Rp'] ** 2]])
        self.Sdsp_B = np.array([[0], [0], [0], [1]])
        self.Sdsp_C = coefDsp * np.array([38115 * self.param['Ds_p'] ** 2, 1980 * self.param['Ds_p'] * self.param['Rp'] ** 2, 43 * self.param['Rp'] ** 4, 0])
        self.Sdsp_D = np.array([0])

        [n, m] = np.shape(self.Sdsp_A)
        self.Sdsp_A_dp = np.eye(n) + self.Sdsp_A * Ts
        self.Sdsp_B_dp = self.Sdsp_B * Ts
        self.Sdsp_C_dp = self.Sdsp_C
        self.Sdsp_D_dp = self.Sdsp_D

        # sensitivity realization in time domain for D_sn from third order pade
        coefDsn = (63 * self.param['Rn']) / (self.param['F'] * self.param['as_n'] * self.param['Ar_n'] * self.param['Ln'] * self.param['Rn'] ** 8)

        self.Sdsn_A = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                           [-(12006225 * self.param['Ds_n'] ** 4) / self.param['Rn'] ** 8, -1309770 * self.param['Ds_n'] ** 3 / self.param['Rn'] ** 6,
                            -42651 * self.param['Ds_n'] ** 2 / self.param['Rn'] ** 4, -378 * self.param['Ds_n'] / self.param['Rn'] ** 2]])
        self.Sdsn_B = np.array([[0], [0], [0], [1]])
        self.Sdsn_C = coefDsn * np.array([38115 * self.param['Ds_n'] ** 2, 1980 * self.param['Ds_n'] * self.param['Rn'] ** 2, 43 * self.param['Rn'] ** 4, 0])
        self.Sdsn_D = np.array([0])

        [n, m] = np.shape(self.Sdsn_A)
        self.Sdsn_A_dn = np.eye(n) + self.Sdsn_A * Ts
        self.Sdsn_B_dn = self.Sdsn_B * Ts
        self.Sdsn_C_dn = self.Sdsn_C
        self.Sdsn_D_dn = self.Sdsn_D

    def compute_init_states(self):
        stoi_n, stoi_p = self.compute_Stoich_coef(self.SOC_0)

        # IF no initial state is supplied to the "step" method, treat step as initial step
        xn_old = np.array([[(stoi_n * self.param['cs_max_n']) / (self.param['rfa_n'] * 10395 * (self.param['Ds_n'] ** 2))], [0],
                           [0]])  # stoi_n100 should be changed if the initial soc is not equal to 50 %
        xp_old = np.array([[(stoi_p * self.param['cs_max_p']) / (self.param['rfa_p'] * 10395 * (self.param['Ds_p'] ** 2))], [0],
                           [0]])  # initial positive electrode ion concentration
        xe_old = np.array([[0], [0]])

        bat_states = {"xn": xn_old, "xp": xp_old, "xe": xe_old}

        return bat_states

    def import_custom_parameters(self, new_param_dict):

        if len(self.param) != len(new_param_dict):
            print("New Param Dict is NOT the same dimension as self.param")
            exit(1)
        else:
            key_list = list(self.param)

            self.param = {key_list[i]: new_param_dict[key_list[i]] for i in range(len(self.param))}

        return

    def expand_parameters(self):

        empty_list = []
        param_key_list = list(self.param)
        # print(len(self.param))

        empty_list = [self.param[param_key_list[i]] for i in range(len(self.param))]

        return empty_list

    @staticmethod
    def OCV_Anode(theta):
        # DUALFOIL: MCMB 2528 graphite(Bellcore) 0.01 < x < 0.9
        Uref = 0.194 + 1.5 * np.exp(-120.0 * theta) + 0.0351 * tanh((theta - 0.286) / 0.083) - 0.0045 * tanh(
            (theta - 0.849) / 0.119) - 0.035 * tanh((theta - 0.9233) / 0.05) - 0.0147 * tanh(
            (theta - 0.5) / 0.034) - 0.102 * tanh((theta - 0.194) / 0.142) - 0.022 * tanh(
            (theta - 0.9) / 0.0164) - 0.011 * tanh((theta - 0.124) / 0.0226) + 0.0155 * tanh((theta - 0.105) / 0.029)

        return Uref

    @staticmethod
    def OCV_Cathod(theta):
        Uref = 2.16216 + 0.07645 * tanh(30.834 - 54.4806 * theta) + 2.1581 * tanh(52.294 - 50.294 * theta) - 0.14169 * \
               tanh(11.0923 - 19.8543 * theta) + 0.2051 * tanh(1.4684 - 5.4888 * theta) + 0.2531 * tanh(
            (-theta + 0.56478) / 0.1316) - 0.02167 * tanh((theta - 0.525) / 0.006)

        return Uref

    def compute_Stoich_coef(self, state_of_charge):
        """
        Compute Stoichiometry Coefficients (ratio of surf. Conc to max conc.) from SOC value via Interpolation
        """
        alpha = state_of_charge

        stoi_n = (self.param['stoi_n100'] - self.param['stoi_n0']) * alpha + self.param['stoi_n0']  # Negative Electrode Interpolant
        stoi_p = self.param['stoi_p0'] - (self.param['stoi_p0'] - self.param['stoi_p100']) * alpha  # Positive Electrode Interpolant
        return [stoi_n, stoi_p]

    def compute_SOC(self, theta_n, theta_p):
        """
        Computes the value of the SOC from either (N or P) electrode given the current
        Stoichiometry Number (Ratio of Surface Conc. to Max Surface Conc. )
        """
        SOC_n = ((theta_n - self.param['stoi_n0'])/(self.param['stoi_n100'] - self.param['stoi_n0']))
        SOC_p = ((theta_p - self.param['stoi_p0'])/(self.param['stoi_p100'] - self.param['stoi_p0']))

        return [SOC_n, SOC_p]

    def OCP_Slope_Cathode(self, theta):
        docvp_dCsep = 0.07645 * (-54.4806 / self.param['cs_max_p']) * ((1.0 / cosh(30.834 - 54.4806 * theta)) ** 2) + 2.1581 * (-50.294 / self.param['cs_max_p']) * ((cosh(52.294 - 50.294 * theta)) ** (-2)) + 0.14169 * (19.854 / self.param['cs_max_p']) * ((cosh(11.0923 - 19.8543 * theta)) ** (-2)) - 0.2051 * (5.4888 / self.param['cs_max_p']) * ((cosh(1.4684 - 5.4888 * theta)) ** (-2)) - 0.2531 / 0.1316 / self.param['cs_max_p'] * ((cosh((-theta + 0.56478) / 0.1316)) ** (-2)) - 0.02167 / 0.006 / self.param['cs_max_p'] * ((cosh((theta - 0.525) / 0.006)) ** (-2))

        return docvp_dCsep

    def OCP_Slope_Anode(self, theta):
        docvn_dCsen = -1.5 * (120.0 / self.param['cs_max_n']) * np.exp(-120.0 * theta) + (0.0351 / (0.083 * self.param['cs_max_n'])) * ((cosh((theta - 0.286) / 0.083)) ** (-2)) - (0.0045 / (self.param['cs_max_n'] * 0.119)) * ((cosh((theta - 0.849) / 0.119)) ** (-2)) - (0.035 / (self.param['cs_max_n'] * 0.05)) * ((cosh((theta - 0.9233) / 0.05)) ** (-2)) - (0.0147 / (self.param['cs_max_n'] * 0.034)) * ((cosh((theta - 0.5) / 0.034)) ** (-2)) - (0.102 / (self.param['cs_max_n'] * 0.142)) * ((cosh((theta - 0.194) / 0.142)) ** (-2)) - (0.022 / (self.param['cs_max_n'] * 0.0164)) * ((cosh((theta - 0.9) / 0.0164)) ** (-2)) - (0.011 / (self.param['cs_max_n'] * 0.0226)) * ((cosh((theta - 0.124) / 0.0226)) ** (-2)) + (0.0155 / (self.param['cs_max_n'] * 0.029)) * ((cosh((theta - 0.105) / 0.029)) ** (-2))

        return docvn_dCsen

    def plot_results(self, xn, xp, xe, yn, yp, yep, theta_n, theta_p, docv_n, docv_p, V_term, time, current, soc, dV_dDsn, dV_dDsp, dCse_dDsn, dCse_dDsp, dV_dEpsi_sn, dV_dEpsi_sp):

        """# plt.subplot(4, 1 ,1)
        plt.figure(0)
        plt.plot(time, yn)
        plt.plot(time, yp)
        plt.ylabel("Surface Concentration")
        plt.xlabel("Time [seconds]")
        plt.title("Time vs Surface Concentration")"""
        #
        plt.figure(1)
        plt.plot(time, V_term)
        plt.ylabel("Terminal Voltage")
        plt.xlabel("Time [seconds}")
        plt.title("Time vs Terminal Voltage")
        #
        plt.figure(2)
        plt.plot(time, current)
        plt.ylabel("Input Current")
        plt.xlabel("Time [seconds}")
        plt.title("Time vs Input Current")

        plt.figure(3)
        plt.plot(time, soc)
        plt.ylabel("State of Charge")
        plt.xlabel("Time (seconds)")
        plt.title("Time vs State of Charge")

        """plt.figure(4)
        plt.plot(time, docv_n, label="D_OCV_P")
        plt.plot(time, docv_p, label="D_OCV_N")
        plt.ylabel("OCV Slope")
        plt.xlabel("Time (seconds)")
        plt.title("Time vs OCV Slope")
        plt.legend(loc="lower left")"""

        plt.figure(5)
        plt.plot(time, docv_p)
        plt.xlabel("Time (seconds)")
        plt.ylabel("OCV Slope 'P' Electrode")
        plt.title("Time vs OCV Slope")

        # plt.figure(6)
        # plt.plot(time, dV_dEpsi_sp*self.param['epsilon_sp'])
        # plt.xlabel("Time (seconds)")
        # plt.ylabel(" Epsilon_SP Sensitivity ")
        # plt.title("Time vs Epsilon_SP Sensitivity")

        plt.figure(6)
        # plt.plot(time, dV_dDsp * self.param['Ds_p'])
        plt.plot(time, dV_dDsp * 1)
        plt.xlabel("Time (seconds)")
        plt.ylabel(" Ds_p Sensitivity ")
        plt.title("Time vs Ds_p Sensitivity")

        plt.figure(7)
        # plt.plot(time, dV_dEpsi_sp * self.param['epsilon_sp'])
        plt.plot(time, dV_dEpsi_sp * self.param['epsilon_sp'])

        # print(current)

        plt.xlabel("Time (seconds)")
        plt.ylabel(" epsilon_sp Sensitivity ")
        plt.title("Time vs epsilon_sp Sensitivity")
        plt.show()

    @staticmethod
    def trim_array(sim_length, valid_length, xn, xp, xe, yn, yp, yep, theta_n, theta_p, docv_n, docv_p, V_term, time, input_cur_prof, soc_list,dV_dDsn, dV_dDsp, dCse_dDsn, dCse_dDsp, dV_dEpsi_sn, dV_dEpsi_sp):

        if sim_length == valid_length:
            return [xn, xp, xe, yn, yp, yep, theta_n, theta_p, docv_n, docv_p, V_term, time, input_cur_prof, soc_list,dV_dDsn, dV_dDsp, dCse_dDsn, dCse_dDsp, dV_dEpsi_sn, dV_dEpsi_sp]
        else:

            xn = xn[:valid_length]
            xp = xp[:valid_length]
            xe = xe[:valid_length]
            yn =yn[ :valid_length]
            yp = yp[: valid_length]
            theta_n = theta_n[:valid_length]
            theta_p = theta_p[:valid_length]
            V_term = V_term[:valid_length]
            time = time[:valid_length]
            input_cur_prof = input_cur_prof[:valid_length]
            soc_list = soc_list[:valid_length]
            docv_n = docv_n[:valid_length]
            docv_p = docv_p[:valid_length]
            dV_dDsn = dV_dDsn[:valid_length]
            dV_dDsp = dV_dDsp[:valid_length]
            dCse_dDsn = dCse_dDsn[:valid_length]
            dCse_dDsp = dCse_dDsp[:valid_length]
            dV_dEpsi_sn = dV_dEpsi_sn[:valid_length]
            dV_dEpsi_sp = dV_dEpsi_sp[:valid_length]

            return [xn, xp, xe, yn, yp, yep, theta_n, theta_p, docv_n, docv_p, V_term, time, input_cur_prof, soc_list, dV_dDsn, dV_dDsp, dCse_dDsn, dCse_dDsp, dV_dEpsi_sn, dV_dEpsi_sp]

    def compute_Sensitivities(self, I, Jn, Jp, j0_n, j0_p, k_n, k_p, theta_n, theta_p, docvn_dCsen, docvp_dCsep, init_sen_state):

        Sepsi_p, Sepsi_n, Sdsp_p, Sdsn_n = init_sen_state["Sepsi_p"], init_sen_state["Sepsi_n"], init_sen_state["Sdsp_p"], init_sen_state["Sdsn_n"]

        theta_p = theta_p*self.param['cs_max_p']
        theta_n = theta_n* self.param['cs_max_n']

        # state space Output Eqn. realization for epsilon_s (Neg & Pos)
        out_Sepsi_p = self.Sepsi_C_dp @ Sepsi_p
        out_Sepsi_n = self.Sepsi_C_dn @ Sepsi_n

        # print(out_Sepsi_p)

        # state space Output Eqn. realization for D_s (neg and Pos)
        out_Sdsp_p = self.Sdsp_C_dp @ Sdsp_p
        out_Sdsn_n = self.Sdsn_C_dn @ Sdsn_n

        # state space realization for epsilon_s (Neg & Pos)
        Sepsi_p_new = self.Sepsi_A_dp @ Sepsi_p + self.Sepsi_B_dp * I  # current input for positive electrode is negative, ... therefore the sensitivity output should be multiplied by -1
        Sepsi_n_new = self.Sepsi_A_dn @ Sepsi_n + self.Sepsi_B_dn * I

        # state space realization for D_s (neg and Pos)
        Sdsp_p_new = self.Sdsp_A_dp @ Sdsp_p + self.Sdsp_B_dp * I
        Sdsn_n_new = self.Sdsn_A_dn @ Sdsn_n + self.Sdsn_B_dn * I

        # rho1p_1 = -np.sign(I) * (-3 * R * T) / (0.5 * F * Rp * as_p) * ((1 + 1 / k_p ** 2) ** (-0.5))
        rho1p = self.param['R'] * self.param['T'] / (0.5 * self.param['F']) * (1 / (k_p + (k_p ** 2 + 1) ** 0.5)) * (1 + k_p / ((k_p ** 2 + 1) ** 0.5)) * (
                -3 * Jp / (2 * self.param['as_p'] ** 2 * j0_p * self.param['Rp']))

        rho2p = (self.param['R'] * self.param['T']) / (2 * 0.5 * self.param['F']) * (self.param['cep'] * self.param['cs_max_p'] - 2 * self.param['cep'] * theta_p) / (
                    self.param['cep'] * theta_p * (self.param['cs_max_p'] - theta_p)) * (1 + (1 / (k_p + .00000001) ** 2)) ** (-0.5)

        # rho1n_1 = np.sign(I) * (-3 * R * T) / (0.5 * F * Rn * as_n) * ((1 + 1 / k_n ** 2) ** (-0.5))
        rho1n = self.param['R'] * self.param['T'] / (0.5 * self.param['F']) * (1 / (k_n + (k_n ** 2 + 1) ** 0.5)) * (1 + k_n / ((k_n ** 2 + 1) ** 0.5)) * (
                -3 * Jn / (2 * self.param['as_n'] ** 2 * j0_n * self.param['Rn']))

        rho2n = (-self.param['R'] * self.param['T']) / (2 * 0.5 * self.param['F']) * (self.param['cen'] * self.param['cs_max_n'] - 2 * self.param['cen'] * theta_n) / (
                    self.param['cen'] * theta_n * (self.param['cs_max_n'] - theta_n)) * (1 + 1 / (k_n + .00000001) ** 2) ** (-0.5)

        # sensitivity of epsilon_sp epsilon_sn
        sen_out_spsi_p = (rho1p + (rho2p + docvp_dCsep)*-out_Sepsi_p)
        sen_out_spsi_n = (rho1n + (rho2n + docvn_dCsen)*out_Sepsi_n)

        out_deta_p_desp = rho1p + rho2p * (-1) * out_Sepsi_p
        out_deta_n_desn = rho1n + rho2n * out_Sepsi_n

        out_semi_linear_p = docvp_dCsep * out_Sepsi_p
        out_semi_linear_n = docvn_dCsen * out_Sepsi_n

        # sensitivity of Dsp Dsn
        sen_out_ds_p = ((rho2p + docvp_dCsep) * (-1 * out_Sdsp_p)) * self.param['Ds_p']
        sen_out_ds_n = ((rho2n + docvn_dCsen) * out_Sdsn_n) * self.param['Ds_n']

        dV_dDsp = sen_out_ds_p
        dV_dDsn = sen_out_ds_n

        dV_dEpsi_sn = sen_out_spsi_n
        dV_dEpsi_sp = sen_out_spsi_p

        dCse_dDsp = -1 * out_Sdsp_p * self.param['Ds_p']
        dCse_dDsn = out_Sdsn_n * self.param['Ds_n']

        # Surface Concentration Sensitivity for Epsilon (pos & neg)
        dCse_dEpsi_sp = -1. * out_Sepsi_p * self.param['epsi_n']
        dCse_dEpsi_sn = out_Sepsi_n * self.param['epsi_n']          # Espi_N and Epsi_p have the same value, Epsi_p currently not defined


        new_sen_states = {"Sepsi_p": Sepsi_p_new, "Sepsi_n": Sepsi_n_new, "Sdsp_p": Sdsp_p_new, "Sdsn_n": Sdsn_n_new}
        new_sen_outputs = {"dV_dDsn": dV_dDsn, "dV_dDsp": dV_dDsp, "dCse_dDsn": dCse_dDsn, "dCse_dDsp": dCse_dDsp, "dV_dEpsi_sn": dV_dEpsi_sn, "dV_dEpsi_sp": dV_dEpsi_sp, 'dCse_dEpsi_sp': dCse_dEpsi_sp, 'dCse_dEpsi_sn': dCse_dEpsi_sn}

        return [new_sen_states, new_sen_outputs]

    def sim(self, init_state=None, zero_init_I=True, I_input=None, CC=True, init_SOC=None, sim_time=None, delta_t=None, trim_results=True, plot_results=False, voltage_limiter=None):
        """
        sim function runs complete solution given a timeseries current profile
        :return: [Terminal Voltage (time series), SOC (time Series) Input Current Profile (time series) ]
        """
        if sim_time is not None:
            self.simulation_time = sim_time
            self.dt = delta_t
            self.num_steps = self.simulation_time//self.dt

        if voltage_limiter is not None:
            self.limit_term_volt = voltage_limiter


        Kup = int(self.num_steps)
        # Populate State Variables with Initial Condition
        xn = np.zeros([3, Kup + 1])         # (Pos & neg) "states"
        xp = np.zeros([3, Kup + 1])
        xe = np.zeros([2, Kup + 1])

        yn = np.zeros(Kup)                  # (Pos & neg) "outputs"
        yp = np.zeros(Kup)
        yep = np.zeros([2, Kup])

        theta_n = np.zeros(Kup)             # (pos & neg) Stoichiometry Ratio
        theta_p = np.zeros(Kup)
        V_term = np.zeros(Kup)              # Terminal Voltage

        time = np.zeros(Kup)
        input_cur_prof = np.zeros(Kup)
        soc_list = np.zeros(Kup)

        docv_dCse_n = np.zeros(Kup)  # (pos & neg) Stoichiometry Ratio
        docv_dCse_p = np.zeros(Kup)

        dV_dDsn = np.zeros(Kup)
        dV_dDsp = np.zeros(Kup)
        dCse_dDsn = np.zeros(Kup)
        dCse_dDsp = np.zeros(Kup)
        dV_dEpsi_sn = np.zeros(Kup)
        dV_dEpsi_sp = np.zeros(Kup)

        # Set Initial Simulation (Step0) Parameters/Inputs
        if CC is True and I_input is None:
            # When CC is True and No value is supplied (assumed) input = default current value
            input_current = self.default_current*np.ones(Kup)

        elif CC is True and I_input is not None and len(I_input) == 1:
            # When CC is True and Current Input provided (assumed Scalar) input = UserDef current value
            input_current = I_input*np.ones(Kup)        # If Constant Current Flag is set True

        elif CC is True and I_input is not None and len(I_input) > 1:
            raise Exception("INVALID input assigned to class.sim() parameter I_input. When CC flag is 'True', I_input ONLY excepts a list of length 1")

        else:
            # When CC is False use User defined list as input current value
            input_current = I_input                     # IF CC "False" custom input profile (assumed correct Length)

        # input_state = init_state            # Pass Initial value of State to the "step()" method for handling
        input_state = self.full_init_state    # Pass Initial value of State to the "step()" method for handling


        # Main Simulation Loop
        for k in range(0, Kup):
            if zero_init_I and k == 0:
                input_current[0] = 0
            else:
                input_current = input_current
            # Perform one iteration of simulation using "step" method
            bat_states, sen_states, outputs, sen_outputs, soc_new, V_out, theta, docv_dCse, done = self.SPMe_step(input_state, input_current[k], full_sim=True)

            # Record Desired values for post-simulation plotting/analysis
            xn[:, [k]], xp[:, [k]], xe[:, [k]] = bat_states["xn"], bat_states["xp"], bat_states["xe"]
            yn[[k]], yp[[k]], yep[:, [k]] = outputs["yn"], outputs["yp"], outputs["yep"]
            theta_n[k], theta_p[k], docv_dCse_n[k], docv_dCse_p[k] = theta[0].item(), theta[1].item(), docv_dCse[0], docv_dCse[1]

            V_term[k], time[k], input_cur_prof[k], soc_list[k] = V_out.item(), self.dt * k, input_current[k], soc_new[1].item()

            dV_dDsn[k], dV_dDsp[k], dCse_dDsn[k], dCse_dDsp[k], dV_dEpsi_sn[k], dV_dEpsi_sp[k] = sen_outputs["dV_dDsn"], sen_outputs["dV_dDsp"], sen_outputs["dCse_dDsn"], sen_outputs["dCse_dDsp"], sen_outputs["dV_dEpsi_sn"], sen_outputs["dV_dEpsi_sp"]

            # print(docv_dCse_n[k])

            if done:
                val_len = k
                break

            if V_term[k] <= 2.75 and trim_results is True:
                val_len = k
                break

            else:
                val_len = self.num_steps

            # Update "step"s inputs to continue and update the simulation
            input_state, init_SOC = [bat_states, sen_states], soc_new

        xn, xp, xe, yn, yp, yep, theta_n, theta_p, docv_dCse_n, docv_dCse_p, V_term, time, input_cur_prof, soc_list, dV_dDsn, dV_dDsp, dCse_dDsn, dCse_dDsp, dV_dEpsi_sn, dV_dEpsi_sp = self.trim_array(self.num_steps, val_len, xn, xp, xe, yn, yp, yep, theta_n, theta_p, docv_dCse_n, docv_dCse_p, V_term, time, input_cur_prof, soc_list, dV_dDsn, dV_dDsp, dCse_dDsn, dCse_dDsp, dV_dEpsi_sn, dV_dEpsi_sp)

        if plot_results:
            self.plot_results(xn, xp, xe, yn, yp, yep, theta_n, theta_p, docv_dCse_n, docv_dCse_p, V_term, time, input_cur_prof, soc_list, dV_dDsn, dV_dDsp, dCse_dDsn, dCse_dDsp, dV_dEpsi_sn, dV_dEpsi_sp )

        return [xn, xp, xe, yn, yp, yep, theta_n, theta_p, docv_dCse_n, docv_dCse_p, V_term, time, input_cur_prof, soc_list, dV_dDsn, dV_dDsp, dCse_dDsn, dCse_dDsp, dV_dEpsi_sn, dV_dEpsi_sp]

    def SPMe_step(self, states=None, I_input=None, full_sim=False):
        """
        step function runs one iteration of the model given the input current and returns output states and quantities
        States: dict(), I_input: scalar, state_of_charge: scalar

        """
        if states is None:
            raise Exception("System States are of type NONE!")

        # Declare INITIAL state variables
        init_bat_states = states[0]
        init_sensitivity_states = states[1]

        # Unpack Initial battery state variables from dict for use in state space computation
        xn_old = init_bat_states['xn']
        xp_old = init_bat_states['xp']
        xe_old = init_bat_states['xe']

        # Declare New state variables to be "overwritten" by output of the state space computation
        bat_states = init_bat_states
        sensitivity_states = init_sensitivity_states

        # Declare state space outputs
        outputs = {"yn": None, "yp": None, "yep": None}
        sensitivity_outputs = {"dV_dDsn": None, "dV_dDsp": None, "dCse_dDsn": None, "dCse_dDsp": None, "dV_dEpsi_sn": None, "dV_dEpsi_sp": None}

        # Set DONE flag to false: NOTE - done flag indicates whether model encountered invalid state. This flag is exposed
        # to the user via the output and is used to allow the "step" method to terminate higher level functionality/simulations.
        done_flag = False

        # Declare LOCAL copy of Battery parameters
        [epsilon_sn, epsilon_sp, epsilon_e_n, epsilon_e_p, F, Rn,
         Rp, R, T, Ar_n, Ar_p, Ln, Lp, Lsep, Lc, Ds_n, Ds_p, De,
         De_p, De_n, kn, kp, stoi_n0, stoi_n100, stoi_p0, stoi_p100,
         SOC, cs_max_n, cs_max_p, Rf, as_n, as_p, Vn, Vp, t_plus,
         cep, cen, rfa_n, rfa_p, epsi_sep, epsi_e, epsi_n, gamak,
         kappa, kappa_eff, kappa_eff_sep] = self.expand_parameters()

        # Create Local Copy of Discrete SS Matrices for Ease of notation when writing Eqns.
        A_dp = self.A_dp
        B_dp = self.B_dp
        C_dp = self.C_dp
        D_dp = self.D_dp

        A_dn = self.A_dn
        B_dn = self.B_dn
        C_dn = self.C_dn
        D_dn = self.D_dn

        Ae_dp = self.Ae_dp
        Be_dp = self.Be_dp
        Ce_dp = self.Ce_dp
        De_dp = self.De_dp

        # If FULL SIM is set True: Shortciruit SIM "I" & "SOC" values into step model (Does not Check for None inputs or default values)
        if full_sim is True:
            I = I_input

            if I == 0:
                # I = .000000001
                I = 0.0
        else:
            # Initialize Input Current
            if I_input is None:
                I = self.default_current     # If no input signal is provided use CC @ default input value
            else:
                I = I_input

        # Molar Current Flux Density (Assumed UNIFORM for SPM)
        Jn = I / Vn
        Jp = -I / Vp

        if Jn == 0:
            print("Molar Current Density (Jn) is equal to zero. This causes 'division by zero' later")
            print("I", I)

        # Compute "current timestep" Concentration from "Battery States" via Output Eqn (Pos & Neg)
        yn_new = C_dn @ xn_old + D_dn * 0
        yp_new = C_dp @ xp_old + D_dp * 0
        yep_new = Ce_dp @ xe_old + De_dp * 0

        outputs["yn"], outputs["yp"], outputs["yep"] = yn_new, yp_new, yep_new

        # Compute "NEXT" time step "Battery States" via State Space Models (Pos & Neg)
        xn_new = A_dn @ xn_old + B_dn * Jn
        xp_new = A_dp @ xp_old + B_dp * Jp
        xe_new = Ae_dp @ xe_old + Be_dp * I

        bat_states["xn"], bat_states["xp"], bat_states["xe"] = xn_new, xp_new, xe_new

        # Electrolyte Dynamics
        vel = (-I * (0.5 * Lp + 0.5 * Ln) / (Ar_n * kappa_eff) + (-I * Lsep) / (Ar_n * kappa_eff_sep) + (2 * R * T * (1 - t_plus) * (1 + 1.2383) * np.log((1000 + yep_new[0].item()) / (1000 + yep_new[1].item()))) / F)  # yep(1, k) = positive boundary;

        # R_e = -I * (0.5 * Lp + 0.5 * Ln) / (Ar_n * kappa_eff) + (-I * Lsep) / (Ar_n * kappa_eff_sep)
        # V_con = (2 * R * T * (1 - t_plus) * (1 + 1.2383) * np.log((1000 + yep_new[0].item()) / (1000 + yep_new[1].item()))) / F
        # phi_n = 0
        # phi_p = phi_n + vel

        # Compute "Exchange Current Density" per Electrode (Pos & Neg)
        i_0n = kn * F * (cen * yn_new * (cs_max_n - yn_new)) ** .5
        i_0p = kp * F * (cep * yp_new * (cs_max_p - yp_new)) ** .5

        # Kappa (pos & Neg)
        k_n = Jn / (2 * as_n * i_0n)
        k_p = Jp / (2 * as_p * i_0p)

        # Compute Electrode "Overpotentials"
        eta_n = (R*T*np.log(k_n + (k_n**2 + 1)**0.5))/(F*0.5)
        eta_p = (R*T*np.log(k_p + (k_p**2 + 1)**0.5))/(F*0.5)

        # Record Stoich Ratio (SOC can be computed from this)
        theta_n = yn_new / cs_max_n
        theta_p = yp_new / cs_max_p

        theta = [theta_n, theta_p]   # Stoichiometry Ratio Coefficent
        soc_new = self.compute_SOC(theta_n, theta_p)

        U_n = self.OCV_Anode(theta_n)
        U_p = self.OCV_Cathod(theta_p)

        docv_dCse_n = self.OCP_Slope_Anode(theta_n)
        docv_dCse_p = self.OCP_Slope_Cathode(theta_p)

        new_sen_states, new_sen_outputs = self.compute_Sensitivities(I, Jn, Jp, i_0n, i_0p, k_n, k_p, theta_n, theta_p, docv_dCse_n, docv_dCse_p, sensitivity_states)
        sensitivity_outputs = new_sen_outputs

        self.next_full_state = [bat_states, new_sen_states]

        # dV_dDsn, dV_dDsp, dCse_dDsn, dCse_dDsp, dV_dEpsi_sn, dV_dEpsi_sp = new_sen_outputs["dV_dDsn"], new_sen_outputs["dV_dDsp"], new_sen_outputs["dCse_dDsn"], new_sen_outputs["dCse_dDsp"], new_sen_outputs["dV_dEpsi_sn"], new_sen_outputs["dV_dEpsi_sp"]

        docv_dCse = [docv_dCse_n, docv_dCse_p]

        V_term = (U_p - U_n) + (eta_p - eta_n) + vel - Rf * I / (Ar_n * Ln * as_n)  # terminal voltage
        R_film = -Rf * I / (Ar_n * Ln * as_n)

        if np.isnan(V_term) and False:
            print(" ######################      SURE     IS      ##################")

            print("INPUT CURRENT", I)
            print("yn_new", yn_new)
            print("yp_new", yp_new)
            print("YEP", yep_new)
            print("i_0n", i_0n)
            print("i_0p", i_0p)
            print("k_n", k_n)
            print("k_p", k_p)
            print("eta_p", eta_p)
            print("eta_n", eta_n)
            print("SOC", theta[0])
            print("U_p", U_p)
            print("U_n", U_n)

            print(" --------- Expressions ------------- ")
            print("(cs_max_n - yn_new)", (cs_max_n - yn_new))
            print("(cs_max_p - yp_new)", (cs_max_p - yp_new))
            print("cs_max_n", cs_max_n)
            print("cs_max_p", cs_max_p)


            expr = (cen * yn_new * (cs_max_n - yn_new)) ** .5

            expr1 = (cs_max_n - yn_new)
            expr2 = abs(cen * yn_new * (cs_max_n - yn_new))

            print(f'kn {kn} F {F} cen {cen} yn_new {yn_new} cs_max_n {cs_max_n}, expression {expr} ')
            print(f'Expression 1 {expr1}')
            print(f'Expression 2 {expr2}')
            print(f'Expression 3 {expr2**.5}')



            print("#################################################################################")
            print("Vel", vel)

        # if soc_new[1] < .07 or soc_new[0] < .005 or soc_new[1] > 1 or soc_new[0] > 1 or np.isnan(V_term) is True:
        #     done_flag = True
        #
        #     return [init_bat_states, sensitivity_states, outputs, sensitivity_outputs, soc_new, V_term, theta, docv_dCse, done_flag]
        #
        # else:
        #
        #     return [bat_states, new_sen_states, outputs, sensitivity_outputs, soc_new, V_term, theta, docv_dCse, done_flag]

        return [bat_states, new_sen_states, outputs, sensitivity_outputs, soc_new, V_term, theta, docv_dCse, done_flag]


if __name__ == "__main__":

    SPMe = SingleParticleModelElectrolyte_w_Sensitivity(sim_time=1300, init_soc=0)

    [xn, xp, xe, yn, yp, yep, theta_n, theta_p, docv_dCse_n, docv_dCse_p, V_term,
     time, current, soc, dV_dDsn, dV_dDsp, dCse_dDsn, dCse_dDsp, dV_dEpsi_sn, dV_dEpsi_sp]\
        = SPMe.sim(CC=True, zero_init_I=False, I_input=[-25.67*3], plot_results=False)




    print(f" Minimum SOC={np.min(soc)} : Maximum SOC={np.max(soc)}")

    print(f"Electrode #1  Concentration Minimum={np.min(theta_n)} : Maximum={np.max(theta_n)}")
    print(f"Electrode #2  Concentration Minimum={np.min(theta_p)} : Maximum={np.max(theta_p)}")


















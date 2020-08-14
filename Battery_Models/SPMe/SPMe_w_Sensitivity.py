from SPMeBatteryParams import *
import numpy as np
import matplotlib.pyplot as plt
from math import asinh, tanh, cosh
import csv


class SingleParticleModelElectrolyte_w_Sensitivity:
    def __init__(self, timestep=1, sim_time=3600):
        # Simulation Settings
        self.dt = timestep
        self.simulation_time = sim_time
        self.num_steps = self.simulation_time//self.dt

        # self.time = np.arange(0, self.duration, self.dt)
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
        self.Ap = np.array([[0, 1, 0], [0, 0, 1], [0, -(3465 * (Ds_p ** 2) / Rp ** 4), - (189 * Ds_p / Rp ** 2)]])
        self.Bp = np.array([[0], [0], [-1]])
        self.Cp = rfa_p * np.array([[10395 * Ds_p ** 2, 1260 * Ds_p * Rp ** 2, 21 * Rp ** 4]])
        self.Dp = np.array([0])

        # Positive electrode SS Discretized
        [n_pos, m_pos] = np.shape(self.Ap)
        self.A_dp = np.eye(n_pos) + self.Ap * Ts
        self.B_dp = self.Bp * Ts
        self.C_dp = self.Cp
        self.D_dp = self.Dp

        # Negative electrode three-state state space model for the particle
        self.An = np.array([[0, 1, 0], [0, 0, 1], [0, - (3465 * (Ds_n ** 2) / Rn ** 4), - (189 * Ds_n / Rn ** 2)]])
        self.Bn = np.array([[0], [0], [-1]])
        self.Cn = rfa_n * np.array([[10395 * Ds_n ** 2, 1260 * Ds_n * Rn ** 2, 21 * Rn ** 4]])
        self.Dn = np.array([0])

        # Negative electrode SS Discretized
        [n_neg, m_neg] = np.shape(self.An)
        self.A_dn = np.eye(n_neg) + self.An * Ts
        self.B_dn = self.Bn * Ts
        self.C_dn = self.Cn
        self.D_dn = self.Dn

        # electrolyte  concentration (boundary)
        a_p0 = -(epsi_n ** (3 / 2) + 4 * epsi_sep ** (3 / 2)) / (80000 * De_p * epsi_n ** (3 / 2) * epsi_sep ** (3 / 2))
        b_p0 = (epsi_n ** 2 * epsi_sep + 24 * epsi_n ** 3 + 320 * epsi_sep ** 3 + 160 * epsi_n ** (3 / 2) * epsi_sep ** (3 / 2)) / (19200000000 * (4 * De_p * epsi_n ** (1 / 2) * epsi_sep ** 3 + De_p * epsi_n ** 2 * epsi_sep ** (3 / 2)))

        a_n0 = (epsi_n ** (3 / 2) + 4 * epsi_sep ** (3 / 2)) / (80000 * De * epsi_n ** (3 / 2) * epsi_sep ** (3 / 2))
        b_n0 = (epsi_n ** 2 * epsi_sep + 24 * epsi_n ** 3 + 320 * epsi_sep ** 3 + 160 * epsi_n ** (3 / 2) * epsi_sep ** (3 / 2)) / (19200000000 * (4 * De_n * epsi_n ** (1 / 2) * epsi_sep ** 3 + De_n * epsi_n ** 2 * epsi_sep ** (3 / 2)))

        self.Aep = np.array([[-1 / b_p0, 0], [0, -1 / b_n0]])
        self.Bep = gamak * np.array([[1], [1]])
        self.Cep = np.array([[a_p0 / b_p0, 0], [0, a_n0 / b_n0]])
        self.Dep = np.array([0])

        [n_elec, m] = np.shape(self.Aep)
        self.Ae_dp = np.eye(n_elec) + self.Aep * Ts
        self.Be_dp = self.Bep * Ts
        self.Ce_dp = self.Cep
        self.De_dp = self.Dep

        # Sensitivities
        # sensitivity realization in time domain for epsilon_sp from third order pade(you can refer to my slides)
        coefp = 3 / (F * Rp ** 6 * as_p ** 2 * Ar_p * Lp)
        self.Sepsi_A_p = np.array([[0, 1, 0], [0, 0, 1], [0, -(3465 * Ds_p ** 2) / Rp ** 4, -(189 * Ds_p) / Rp ** 2]])
        self.Sepsi_B_p = np.array([[0], [0], [1]])
        self.Sepsi_C_p = coefp * np.array([10395 * Ds_p ** 2, 1260 * Ds_p * Rp ** 2, 21 * Rp ** 4])
        self.Sepsi_D_p = np.array([0])

        [n, m] = np.shape(self.Sepsi_A_p)
        self.Sepsi_A_dp = np.eye(n) + self.Sepsi_A_p * Ts
        self.Sepsi_B_dp = self.Sepsi_B_p * Ts
        self.Sepsi_C_dp = self.Sepsi_C_p
        self.Sepsi_D_dp = self.Sepsi_D_p

        # sensitivity realization in time domain for epsilon_sn from third order pade(you can refer to my slides)
        coefn = 3 / (F * Rn ** 6 * as_n ** 2 * Ar_n * Ln)

        self.Sepsi_A_n = np.array([[0, 1, 0], [0, 0, 1], [0, -(3465 * Ds_n ** 2) / Rn ** 4, -(189 * Ds_n) / Rn ** 2]])
        self.Sepsi_B_n = np.array([[0], [0], [1]])
        self.Sepsi_C_n = coefn * np.array([10395 * Ds_n ** 2, 1260 * Ds_n * Rn ** 2, 21 * Rn ** 4])
        self.Sepsi_D_n = np.array([0])

        [n, m] = np.shape(self.Sepsi_A_n)
        self.Sepsi_A_dn = np.eye(n) + self.Sepsi_A_n * Ts
        self.Sepsi_B_dn = self.Sepsi_B_n * Ts
        self.Sepsi_C_dn = self.Sepsi_C_n
        self.Sepsi_D_dn = self.Sepsi_D_n

        # sensitivity realization in time domain for D_sp from third order pade
        coefDsp = (63 * Rp) / (F * as_p * Ar_p * Lp * Rp ** 8)

        self.Sdsp_A = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                           [-(12006225 * Ds_p ** 4) / Rp ** 8, -1309770 * Ds_p ** 3 / Rp ** 6,
                            -42651 * Ds_p ** 2 / Rp ** 4, -378 * Ds_p / Rp ** 2]])
        self.Sdsp_B = np.array([[0], [0], [0], [1]])
        self.Sdsp_C = coefDsp * np.array([38115 * Ds_p ** 2, 1980 * Ds_p * Rp ** 2, 43 * Rp ** 4, 0])
        self.Sdsp_D = np.array([0])

        [n, m] = np.shape(self.Sdsp_A)
        self.Sdsp_A_dp = np.eye(n) + self.Sdsp_A * Ts
        self.Sdsp_B_dp = self.Sdsp_B * Ts
        self.Sdsp_C_dp = self.Sdsp_C
        self.Sdsp_D_dp = self.Sdsp_D

        # sensitivity realization in time domain for D_sn from third order pade
        coefDsn = (63 * Rn) / (F * as_n * Ar_n * Ln * Rn ** 8)

        self.Sdsn_A = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                           [-(12006225 * Ds_n ** 4) / Rn ** 8, -1309770 * Ds_n ** 3 / Rn ** 6,
                            -42651 * Ds_n ** 2 / Rn ** 4, -378 * Ds_n / Rn ** 2]])
        self.Sdsn_B = np.array([[0], [0], [0], [1]])
        self.Sdsn_C = coefDsn * np.array([38115 * Ds_n ** 2, 1980 * Ds_n * Rn ** 2, 43 * Rn ** 4, 0])
        self.Sdsn_D = np.array([0])

        [n, m] = np.shape(self.Sdsn_A)
        self.Sdsn_A_dn = np.eye(n) + self.Sdsn_A * Ts
        self.Sdsn_B_dn = self.Sdsn_B * Ts
        self.Sdsn_C_dn = self.Sdsn_C
        self.Sdsn_D_dn = self.Sdsn_D

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

    @staticmethod
    def compute_Stoich_coef(state_of_charge):
        """
        Compute Stoichiometry Coefficients (ratio of surf. Conc to max conc.) from SOC value via Interpolation
        """
        alpha = state_of_charge

        stoi_n = (stoi_n100 - stoi_n0) * alpha + stoi_n0  # Negative Electrode Interpolant
        stoi_p = stoi_p0 - (stoi_p0 - stoi_p100) * alpha  # Positive Electrode Interpolant
        return [stoi_n, stoi_p]

    @staticmethod
    def compute_SOC(theta_n, theta_p):
        """
        Computes the value of the SOC from either (N or P) electrode given the current
        Stoichiometry Number (Ratio of Surface Conc. to Max Surface Conc. )
        """
        SOC_n = ((theta_n - stoi_n0)/(stoi_n100 - stoi_n0))
        SOC_p = ((theta_p - stoi_p0)/(stoi_p100 - stoi_p0))

        return [SOC_n, SOC_p]

    @staticmethod
    def OCP_Slope_Cathode(theta):
        docvp_dCsep = 0.07645 * (-54.4806 / cs_max_p) * ((1.0 / cosh(30.834 - 54.4806 * theta)) ** 2) + 2.1581 * (-50.294 / cs_max_p) * ((cosh(52.294 - 50.294 * theta)) ** (-2)) + 0.14169 * (19.854 / cs_max_p) * ((cosh(11.0923 - 19.8543 * theta)) ** (-2)) - 0.2051 * (5.4888 / cs_max_p) * ((cosh(1.4684 - 5.4888 * theta)) ** (-2)) - 0.2531 / 0.1316 / cs_max_p * ((cosh((-theta + 0.56478) / 0.1316)) ** (-2)) - 0.02167 / 0.006 / cs_max_p * ((cosh((theta - 0.525) / 0.006)) ** (-2))

        return docvp_dCsep

    @staticmethod
    def OCP_Slope_Anode(theta):
        docvn_dCsen = -1.5 * (120.0 / cs_max_n) * np.exp(-120.0 * theta) + (0.0351 / (0.083 * cs_max_n)) * ((cosh((theta - 0.286) / 0.083)) ** (-2)) - (0.0045 / (cs_max_n * 0.119)) * ((cosh((theta - 0.849) / 0.119)) ** (-2)) - (0.035 / (cs_max_n * 0.05)) * ((cosh((theta - 0.9233) / 0.05)) ** (-2)) - (0.0147 / (cs_max_n * 0.034)) * ((cosh((theta - 0.5) / 0.034)) ** (-2)) - (0.102 / (cs_max_n * 0.142)) * ((cosh((theta - 0.194) / 0.142)) ** (-2)) - (0.022 / (cs_max_n * 0.0164)) * ((cosh((theta - 0.9) / 0.0164)) ** (-2)) - (0.011 / (cs_max_n * 0.0226)) * ((cosh((theta - 0.124) / 0.0226)) ** (-2)) + (0.0155 / (cs_max_n * 0.029)) * ((cosh((theta - 0.105) / 0.029)) ** (-2))

        return docvn_dCsen

    @staticmethod
    def plot_results(xn, xp, xe, yn, yp, yep, theta_n, theta_p, docv_n, docv_p, V_term, time, current, soc, dV_dDsn, dV_dDsp, dCse_dDsn, dCse_dDsp, dV_dEpsi_sn, dV_dEpsi_sp):

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

        plt.figure(6)
        plt.plot(time, dV_dEpsi_sp*epsilon_sp)
        plt.xlabel("Time (seconds)")
        plt.ylabel(" Epsilon_SP Sensitivity ")
        plt.title("Time vs Epsilon_SP Sensitivity")
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

        theta_p = theta_p*cs_max_p
        theta_n = theta_n* cs_max_n

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
        rho1p = R * T / (0.5 * F) * (1 / (k_p + (k_p ** 2 + 1) ** 0.5)) * (1 + k_p / ((k_p ** 2 + 1) ** 0.5)) * (
                -3 * Jp / (2 * as_p ** 2 * j0_p * Rp))

        rho2p = (R * T) / (2 * 0.5 * F) * (cep * cs_max_p - 2 * cep * theta_p) / (
                    cep * theta_p * (cs_max_p - theta_p)) * (1 + (1 / (k_p) ** 2)) ** (-0.5)


        # print(rho1p, "|", rho2p )
        # print(theta_p)
        # rho1n_1 = np.sign(I) * (-3 * R * T) / (0.5 * F * Rn * as_n) * ((1 + 1 / k_n ** 2) ** (-0.5))
        rho1n = R * T / (0.5 * F) * (1 / (k_n + (k_n ** 2 + 1) ** 0.5)) * (1 + k_n / ((k_n ** 2 + 1) ** 0.5)) * (
                -3 * Jn / (2 * as_n ** 2 * j0_n * Rn))

        rho2n = (-R * T) / (2 * 0.5 * F) * (cen * cs_max_n - 2 * cen * theta_n) / (
                    cen * theta_n * (cs_max_n - theta_n)) * (1 + 1 / (k_n) ** 2) ** (-0.5)

        # sensitivity of epsilon_sp epsilon_sn
        sen_out_spsi_p = (rho1p + (rho2p + docvp_dCsep)*-out_Sepsi_p)
        sen_out_spsi_n = (rho1n + (rho2n + docvn_dCsen)*out_Sepsi_n)

        out_deta_p_desp = rho1p + rho2p * (-1) * out_Sepsi_p
        out_deta_n_desn = rho1n + rho2n * out_Sepsi_n

        out_semi_linear_p = docvp_dCsep * out_Sepsi_p
        out_semi_linear_n = docvn_dCsen * out_Sepsi_n

        # sensitivity of Dsp Dsn
        sen_out_ds_p = ((rho2p + docvp_dCsep) * (-1 * out_Sdsp_p)) * Ds_p
        sen_out_ds_n = ((rho2n + docvn_dCsen) * out_Sdsn_n) * Ds_n

        dV_dDsp = sen_out_ds_p
        dV_dDsn = sen_out_ds_n

        dV_dEpsi_sn = sen_out_spsi_n
        dV_dEpsi_sp = sen_out_spsi_p

        dCse_dDsp = -1 * out_Sdsp_p * Ds_p
        dCse_dDsn = out_Sdsn_n * Ds_n

        new_sen_states = {"Sepsi_p": Sepsi_p_new, "Sepsi_n": Sepsi_n_new, "Sdsp_p": Sdsp_p_new, "Sdsn_n": Sdsn_n_new}
        new_sen_outputs = {"dV_dDsn": dV_dDsn, "dV_dDsp": dV_dDsp, "dCse_dDsn": dCse_dDsn, "dCse_dDsp": dCse_dDsp, "dV_dEpsi_sn": dV_dEpsi_sn, "dV_dEpsi_sp": dV_dEpsi_sp}

        return [new_sen_states, new_sen_outputs]

    def sim(self, init_state=None, zero_init_I=True, I_input=None, CC=True, init_SOC=None, sim_time=None, delta_t=None, plot_results=False):
        """
        sim function runs complete solution given a timeseries current profile
        :return: [Terminal Voltage (time series), SOC (time Series) Input Current Profile (time series) ]
        """
        if sim_time is not None:
            self.simulation_time = sim_time
            self.dt = delta_t
            self.num_steps = self.simulation_time//self.dt

        Kup = self.num_steps
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

        input_state = init_state            # Pass Initial value of State to the "step()" method for handling

        # Main Simulation Loop
        for k in range(0, Kup):
            if zero_init_I and k == 0:
                input_current[0] = 0
            else:
                input_current = input_current
            # Perform one iteration of simulation using "step" method
            bat_states, sen_states, outputs, sen_outputs, soc_new, V_out, theta, docv_dCse = self.step(input_state, input_current[k], init_SOC, full_sim=True)

            # Record Desired values for post-simulation plotting/analysis
            xn[:, [k]], xp[:, [k]], xe[:, [k]] = bat_states["xn"], bat_states["xp"], bat_states["xe"]
            yn[[k]], yp[[k]], yep[:, [k]] = outputs["yn"], outputs["yp"], outputs["yep"]
            theta_n[k], theta_p[k], docv_dCse_n[k], docv_dCse_p[k] = theta[0].item(), theta[1].item(), docv_dCse[0], docv_dCse[1]

            V_term[k], time[k], input_cur_prof[k], soc_list[k] = V_out.item(), self.dt * k, input_current[k], soc_new[0].item()

            dV_dDsn[k], dV_dDsp[k], dCse_dDsn[k], dCse_dDsp[k], dV_dEpsi_sn[k], dV_dEpsi_sp[k] = sen_outputs["dV_dDsn"], sen_outputs["dV_dDsp"], sen_outputs["dCse_dDsn"], sen_outputs["dCse_dDsp"], sen_outputs["dV_dEpsi_sn"], sen_outputs["dV_dEpsi_sp"]

            # print(docv_dCse_n[k])

            if V_term[k] <= 2.75:
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

    def step(self, states=None, I_input=None, state_of_charge=None, full_sim=False):
        """
        step function runs one iteration of the model given the input current and returns output states and quantities
        States: dict(), I_input: scalar, state_of_charge: scalar
        """
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
            soc = state_of_charge
        else:
            # Initialize Input Current
            if I_input is None:
                I = self.default_current     # If no input signal is provided use CC @ default input value
            else:
                I = I_input

            # Initialize SOC
            if state_of_charge is None:
                soc = .5                    # If no SOC is provided by user then defaults to SOC = .5
            else:
                soc = state_of_charge

        # Initialize "State" Vector
        if states is None:
            stoi_n, stoi_p = self.compute_Stoich_coef(soc)

            # IF not initial state is supplied to the "step" method, treat step as initial step
            xn_old = np.array([[(stoi_n * cs_max_n) / (rfa_n * 10395 * (Ds_n ** 2))], [0], [0]])  # stoi_n100 should be changed if the initial soc is not equal to 50 %
            xp_old = np.array([[(stoi_p * cs_max_p) / (rfa_p * 10395 * (Ds_p ** 2))], [0], [0]])  # initial positive electrode ion concentration
            xe_old = np.array([[0], [0]])

            Sepsi_p_old = np.array([[0], [0], [0]])
            Sepsi_n_old = np.array([[0], [0], [0]])
            Sdsp_p_old = np.array([[0], [0], [0], [0]])
            Sdsn_n_old = np.array([[0], [0], [0], [0]])

            bat_states = {"xn": xn_old, "xp": xp_old, "xe": xe_old}
            outputs = {"yn": None, "yp": None, "yep": None}

            sensitivity_states = {"Sepsi_p": Sepsi_p_old, "Sepsi_n": Sepsi_n_old, "Sdsp_p": Sdsp_p_old, "Sdsn_n": Sdsn_n_old}
            sensitivity_outputs = {"dV_dDsn": None, "dV_dDsp": None, "dCse_dDsn": None, "dCse_dDsp": None, "dV_dEpsi_sn": None, "dV_dEpsi_sp": None}

        else:
            bat_states = states[0]
            init_bat_states = bat_states
            init_sen_states = states[1]

            # ELSE use given states information to propagate model forward in time
            xn_old, xp_old, xe_old = bat_states["xn"], bat_states["xp"], bat_states["xe"]
            outputs = {"yn": None, "yp": None, "yep": None}

            sensitivity_states = init_sen_states
            sensitivity_outputs = {"dV_dDsn": None, "dV_dDsp": None, "dCse_dDsn": None, "dCse_dDsp": None, "dV_dEpsi_sn": None, "dV_dEpsi_sp": None}

        # Molar Current Flux Density (Assumed UNIFORM for SPM)
        Jn = I / Vn
        Jp = -I / Vp

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
        R_e = -I * (0.5 * Lp + 0.5 * Ln) / (Ar_n * kappa_eff) + (-I * Lsep) / (Ar_n * kappa_eff_sep)
        V_con = (2 * R * T * (1 - t_plus) * (1 + 1.2383) * np.log((1000 + yep_new[0].item()) / (1000 + yep_new[1].item()))) / F
        phi_n = 0
        phi_p = phi_n + vel

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

        # dV_dDsn, dV_dDsp, dCse_dDsn, dCse_dDsp, dV_dEpsi_sn, dV_dEpsi_sp = new_sen_outputs["dV_dDsn"], new_sen_outputs["dV_dDsp"], new_sen_outputs["dCse_dDsn"], new_sen_outputs["dCse_dDsp"], new_sen_outputs["dV_dEpsi_sn"], new_sen_outputs["dV_dEpsi_sp"]

        docv_dCse = [docv_dCse_n, docv_dCse_p]

        V_term = (U_p - U_n) + (eta_p - eta_n) + vel - Rf * I / (Ar_n * Ln * as_n)  # terminal voltage
        R_film = -Rf * I / (Ar_n * Ln * as_n)

        if V_term <= 2.75: # or V_term >= 4.2:
            return [init_bat_states, sensitivity_states, outputs, sensitivity_outputs, soc_new, V_term, theta, docv_dCse]
        else:
            return [bat_states, new_sen_states, outputs, sensitivity_outputs, soc_new, V_term, theta, docv_dCse]

if __name__ == "__main__":

    SPMe = SingleParticleModelElectrolyte_w_Sensitivity(sim_time=1300)

    [xn, xp, xe, yn, yp, yep, theta_n, theta_p, docv_dCse_n, docv_dCse_p, V_term,
     time, current, soc, dV_dDsn, dV_dDsp, dCse_dDsn, dCse_dDsp, dV_dEpsi_sn, dV_dEpsi_sp]\
        = SPMe.sim(CC=True, zero_init_I=True, I_input=[-25.67*3], init_SOC=0, plot_results=True)


















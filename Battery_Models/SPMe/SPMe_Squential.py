from SPMeBatteryParams import *
import numpy as np
import matplotlib.pyplot as plt
from numpy import tanh


def OCV_Anode(theta):
    # DUALFOIL: MCMB 2528 graphite(Bellcore) 0.01 < x < 0.9
    Uref = 0.194 + 1.5 * np.exp(-120.0 * theta) + 0.0351 * tanh((theta - 0.286) / 0.083) - 0.0045 * tanh(
        (theta - 0.849) / 0.119) - 0.035 * tanh((theta - 0.9233) / 0.05) - 0.0147 * tanh(
        (theta - 0.5) / 0.034) - 0.102 * tanh((theta - 0.194) / 0.142) - 0.022 * tanh(
        (theta - 0.9) / 0.0164) - 0.011 * tanh((theta - 0.124) / 0.0226) + 0.0155 * tanh((theta - 0.105) / 0.029)

    return Uref

def OCV_Cathod(theta):
    Uref = 2.16216 + 0.07645 * tanh(30.834 - 54.4806 * theta) + 2.1581 * tanh(52.294 - 50.294 * theta) - 0.14169 * \
           tanh(11.0923 - 19.8543 * theta) + 0.2051 * tanh(1.4684 - 5.4888 * theta) + 0.2531 * tanh((-theta + 0.56478)\
           / 0.1316) - 0.02167 * tanh((theta - 0.525) / 0.006)

    return Uref

def compute_Stoich_coef(state_of_charge):
    """
    Compute Stoichiometry Coefficients (ratio of surf. Conc to max conc.) from SOC value via Interpolation
    """
    alpha = state_of_charge

    stoi_n = (stoi_n100 - stoi_n0) * alpha + stoi_n0  # Negative Electrode Interpolant
    stoi_p = stoi_p0 - (stoi_p0 - stoi_p100) * alpha  # Positive Electrode Interpolant
    return [stoi_n, stoi_p]

def compute_SOC(theta_n, theta_p):
    """
    Computes the value of the SOC from either (N or P) electrode given the current
    Stoichiometry Number (Ratio of Surface Conc. to Max Surface Conc. )
    """
    SOC_n = ((theta_n - stoi_n0)/(stoi_n100 - stoi_n0))
    SOC_p = ((theta_p - stoi_p0)/(stoi_p100 - stoi_p0))

    return [SOC_n, SOC_p]

def a_p0():

    a_p0 = -(epsi_n ** (1.5) + 4 * epsi_sep ** (1.5)) / (8e4 * De_p * epsi_n ** (1.5) * epsi_sep ** (1.5))


    return a_p0

def b_p0():
    b_p0 = (epsi_n ** 2 * epsi_sep + 24 * epsi_n ** 3 + 320 * epsi_sep ** 3 + 160 * epsi_n ** (1.5) * epsi_sep ** (
            1.5)) / (1.92e10 * (
            4 * De_p * epsi_n ** (.5) * epsi_sep ** 3 + De_p * epsi_n ** 2 * epsi_sep ** (1.5)))

    return b_p0

def a_n0():
    a_n0 = (epsi_n ** (1.5) + 4 * epsi_sep ** (1.5)) / (8e4 * De * epsi_n ** (1.5) * epsi_sep ** (1.5))

    return a_n0

def b_n0():
    b_n0 = (epsi_n ** 2 * epsi_sep + 24 * epsi_n ** 3 + 320 * epsi_sep ** 3 + 160 * epsi_n ** (1.5) * epsi_sep ** (
            1.5)) / (1.92e10 * (
            4 * De_n * epsi_n ** (.5) * epsi_sep ** 3 + De_n * epsi_n ** 2 * epsi_sep ** (1.5)))

    return b_n0


if __name__ =="__main__":

    num_steps = int(3600/1)
    Ts = 1

    # Model Parameters & Variables
    ###################################################################
    # Positive electrode three-state state space model for the particle
    Ap = np.array([[0, 1, 0], [0, 0, 1], [0, -(3465 * (Ds_p ** 2) / Rp ** 4), - (189 * Ds_p / Rp ** 2)]])
    Bp = np.array([[0], [0], [-1]])
    Cp = rfa_p * np.array([[10395 * Ds_p ** 2, 1260 * Ds_p * Rp ** 2, 21 * Rp ** 4]])
    Dp = np.array([0])

    # Positive electrode SS Discretized
    [n_pos, _] = np.shape(Ap)
    A_dp = np.eye(n_pos) + Ap * Ts
    B_dp = Bp * Ts
    C_dp = Cp
    D_dp = Dp

    # Negative electrode three-state state space model for the particle
    An = np.array([[0, 1, 0], [0, 0, 1], [0, - (3465 * (Ds_n ** 2) / Rn ** 4), - (189 * Ds_n / Rn ** 2)]])
    Bn = np.array([[0], [0], [-1]])
    Cn = rfa_n * np.array([[10395 * Ds_n ** 2, 1260 * Ds_n * Rn ** 2, 21 * Rn ** 4]])
    Dn = np.array([0])

    # Negative electrode SS Discretized
    [n_neg, _] = np.shape(An)
    A_dn = np.eye(n_neg) + An * Ts
    B_dn = Bn * Ts
    C_dn = Cn
    D_dn = Dn

    # electrolyte  concentration (boundary)
    a_p0 = a_p0()
    b_p0 = b_p0()

    a_n0 = a_n0()
    b_n0 = b_n0()

    Aep = np.array([[(-1. / b_p0), 0.], [0., (-1. / b_n0)]])
    Bep = gamak * np.array([[1.], [1.]])
    Cep = np.array([[(a_p0 / b_p0), 0.], [0., (a_n0 / b_n0)]])
    Dep = np.array([0.])

    [n_elec, _] = np.shape(Aep)
    Ae_dp = np.eye(n_elec) + Aep * Ts
    Be_dp = Bep * Ts
    Ce_dp = Cep
    De_dp = Dep

    Kup = num_steps

    # Model Initial Conditions
    ###################################################################
    # Populate State Variables with Initial Condition
    xn = np.zeros([3, Kup + 1])  # (Pos & neg) "states"
    xp = np.zeros([3, Kup + 1])
    xe = np.zeros([2, Kup + 1])

    yn = np.zeros(Kup)  # (Pos & neg) "outputs"
    yp = np.zeros(Kup)
    yep = np.zeros([2, Kup])

    theta_n = np.zeros(Kup)  # (pos & neg) Stoichiometry Ratio
    theta_p = np.zeros(Kup)
    V = np.zeros(Kup)  # Terminal Voltage

    time = np.zeros(Kup)
    input_cur_prof = np.zeros(Kup)
    soc_list = np.zeros(Kup)

    current_prof = 25.67*np.ones(Kup)
    current_prof[0] = 0
    soc = .5
    states = None

    # Main Simulation Loop
    for k in range(0, Kup):
        I = current_prof[k]

        # Initialize "State" Vector
        if states is None:
            stoi_n, stoi_p = compute_Stoich_coef(soc)

            # IF not initial state is supplied to the "step" method, treat step as initial step
            xn_old = np.array([[(stoi_n * cs_max_n) / (rfa_n * 10395 * (Ds_n ** 2))], [0], [0]])
            xp_old = np.array([[(stoi_p * cs_max_p) / (rfa_p * 10395 * (Ds_p ** 2))], [0], [0]])
            xe_old = np.array([[0], [0]])

            states = {"xn": xn_old, "xp": xp_old, "xe": xe_old}
            outputs = {"yn": None, "yp": None, "yep": np.array([[0], [0]])}

        else:
            # ELSE use given states information to propagate model forward in time
            xn_old, xp_old, xe_old = states["xn"], states["xp"], states["xe"]
            outputs = {"yn": None, "yp": None, "yep": None}

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

        states["xn"], states["xp"], states["xe"] = xn_new, xp_new, xe_new

        # Electrolyte Dynamics

        delta_phi_e_con = -I * (0.5 * Lp + 0.5 * Ln) / (Ar_n * kappa_eff) + (-I * Lsep) / (Ar_n * kappa_eff_sep)
        delta_phi_e_omega = (2 * R * T * (1 - t_plus) * (1 + 1.2383) * np.log((1000 + yep_new[0].item()) / (1000 + yep_new[1].item()))) / F
        # vel = (-I * (0.5 * Lp + 0.5 * Ln) / (Ar_n * kappa_eff) + (-I * Lsep) / (Ar_n * kappa_eff_sep) + (2 * R * T * (1 - t_plus) * (1 + 1.2383) * np.log((1000 + yep_new[0].item()) / (1000 + yep_new[1].item()))) / F)  # yep(1, k) = positive boundary;
        vel = delta_phi_e_omega + delta_phi_e_con

        phi_n = 0
        phi_p = phi_n + vel

        # Compute "Exchange Current Density" per Electrode (Pos & Neg)
        i_0n = kn * F * (cen * yn_new * (cs_max_n - yn_new)) ** .5
        i_0p = kp * F * (cep * yp_new * (cs_max_p - yp_new)) ** .5

        # Kappa (pos & Neg)
        k_n = Jn / (2 * as_n * i_0n)
        k_p = Jp / (2 * as_p * i_0p)

        # Compute Electrode "Overpotentials"
        eta_n = (R * T * np.log(k_n + (k_n ** 2 + 1) ** 0.5)) / (F * 0.5)
        eta_p = (R * T * np.log(k_p + (k_p ** 2 + 1) ** 0.5)) / (F * 0.5)

        # Record Stoich Ratio (SOC can be computed from this)
        theta_n_scalar = yn_new / cs_max_n
        theta_p_scalar = yp_new / cs_max_p

        # theta = [theta_n, theta_p]  # Stoichiometry Ratio Coefficent
        soc_new = compute_SOC(theta_n_scalar, theta_p_scalar)

        eata = eta_p - eta_n

        U_n = OCV_Anode(theta_n_scalar)
        U_p = OCV_Cathod(theta_p_scalar)

        # V_term = U_p - U_n + eta_p - eta_n
        V_term = (U_p - U_n) + eata + vel - Rf * I / (Ar_n * Ln * as_n)  # terminal voltage
        R_film = -Rf * I / (Ar_n * Ln * as_n)

        if V_term <= 2.75:
            break

        # Record Desired values for post-simulation plotting/analysis
        xn[:, [k]], xp[:, [k]], xe[:, [k]], theta_n[k], theta_p[k] = states["xn"], states["xp"], states["xe"], theta_n_scalar, theta_p_scalar
        yn[[k]], yp[[k]], yep[:, [k]] = outputs["yn"], outputs["yp"], outputs["yep"]

        V[k], time[k], input_cur_prof[k], soc_list[k] = V_term.item(), Ts * k, I, soc_new[0].item()



        # Update "step"s inputs to continue and update the simulation
        input_state, init_SOC = states, soc_new


    counter = 0
    for i in range(0, len(V)):
        if V[i] > 2.75:
            counter += 1
    print("Counter", counter)

    V_trunc = V[:counter]
    SOC_trunc = soc_list[:counter]
    current = input_cur_prof[:counter]

    print(len(V_trunc))



    plt.figure(0)
    plt.plot(SOC_trunc)
    #
    plt.figure(1)
    plt.plot(V_trunc)
    #
    plt.figure()
    plt.plot(current)
    plt.show()


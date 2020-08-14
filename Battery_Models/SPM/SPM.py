from SPMBatteryParams import *
import numpy as np
import matplotlib.pyplot as plt


from math import asinh, tanh


def OCV_Anode(theta):
    # DUALFOIL: MCMB 2528 graphite(Bellcore) 0.01 < x < 0.9
    Uref = 0.194 + 1.5 * np.exp(-120.0 * theta)
    + 0.0351 * tanh((theta - 0.286) / 0.083)
    - 0.0045 * tanh((theta - 0.849) / 0.119)
    - 0.035 * tanh((theta - 0.9233) / 0.05)
    - 0.0147 * tanh((theta - 0.5) / 0.034)
    - 0.102 * tanh((theta - 0.194) / 0.142)
    - 0.022 * tanh((theta - 0.9) / 0.0164)
    - 0.011 * tanh((theta - 0.124) / 0.0226)
    + 0.0155 * tanh((theta - 0.105) / 0.029)

    return Uref


def OCV_Cathod(theta):
    Uref = 2.16216 + 0.07645 * tanh(30.834 - 54.4806 * theta)
    + 2.1581 * tanh(52.294 - 50.294 * theta)
    - 0.14169 * tanh(11.0923 - 19.8543 * theta)
    + 0.2051 * tanh(1.4684 - 5.4888 * theta)
    + 0.2531 * tanh((-theta + 0.56478) / 0.1316)
    - 0.02167 * tanh((theta - 0.525) / 0.006)

    return Uref


# Simulation Time
Kup = 1300
# C-Rates (battery Charge/Discharge Rate)
# 3601sec # 1C
# 1712sec # 2C
# 1083sec # 3C              ;

I = np.zeros(Kup)
#  Generate/load pulse profile
for k in range(0, Kup):
    if k == 0:
        I[k] = 0
    else:
        # I(k) = 1
        # I(k) = 25.5;
        I[k] = -25.67 * 3

Kup = len(I)

# Negative electrode three-state state space model for the particle
An = np.array([[0, 1, 0], [0, 0, 1], [0, - (3465 * (Ds_n ** 2) / Rn ** 4), - (189 * Ds_n / Rn ** 2)]])
Bn = np.array([[0], [0], [-1]])
Cn = rfa_n * np.array([[10395 * Ds_n ** 2,  1260 * Ds_n * Rn ** 2,   21 * Rn ** 4]])
Dn = np.array([0])

# Approximate Negative Electrode Discretization
Ts = 1
[n, m] = np.shape(An)
A_dn = np.eye(n) + An * Ts
B_dn = Bn * Ts
C_dn = Cn
D_dn = Dn

# a_n = A_dn
# b_n = B_dn
# c_n = Cn
# d_n = Dn

# discharge

# Populate State Variables with Initial Condition
xn = np.zeros([3, Kup+1])
xp = np.zeros([3, Kup+1])
yn = np.zeros(Kup+1)
yp = np.zeros(Kup+1)
theta_n = np.zeros(Kup)
theta_p = np.zeros(Kup)
V_term = np.zeros(Kup)
time = np.zeros(Kup)


xn[:, [0]] = np.array([[stoi_x * cs_max_n / (rfa_n * 10395 * Ds_n ** 2)], [0], [0]]) # stoi_x100 should be changed if the initial soc is not equal to 50 %
xp[:, [0]] = np.array([[stoi_y * cs_max_p / (rfa_p * 10395 * Ds_p ** 2)], [0], [0]]) # initial positive electrode ion concentration


# Positive electrode three-state state space model for the particle
Ap = 1*np.array([[0, 1, 0], [0, 0, 1], [0, -(3465 * (Ds_p ** 2) / Rp ** 4), - (189 * Ds_p / Rp ** 2)]])
Bp = np.array([[0], [0], [1]])
Cp = rfa_p * np.array([[10395 * Ds_p ** 2, 1260 * Ds_p * Rp ** 2, 21 * Rp ** 4]])
Dp = np.array([0])

# Approximate Positive Electrode Discretization
[n, m] = np.shape(Ap)
A_dp = np.eye(n) + Ap * Ts
B_dp = Bp * Ts
C_dp = Cp
D_dp = Dp




for k in range(0, Kup):
    time[k] = k

    """    # Negative electrode three-state state space model for the particle
    I_Qdyn1[k] = I[k]
    Jn = I[k] / Vn
    ut_n = Jn
    i[k] = k * Ts
    yn[k] = c_n * xn[:, k]
    xn[:, k + 1]=a_n * xn[:, k]+b_n * ut_n

    j0_n[k] = kn * F * ((cen) * (cs_max_n - yn[k]) * yn[k]) ** (0.5)
    k_n[k] = Jn / (2 * as_n * j0_n[k])
    eta_n[k] = R * T * np.log(k_n[k] + (k_n[k] ** 2 + 1) ** 0.5) / (F * 0.5)

    x[k] = yn[k] / (cs_max_n)

    #  Positive electrode three - state state space model for the particle
    Jp = -I[k] / Vp # current densitivity input for cathode is negative
    ut_p = Jp

    j1[k] = k * Ts

    yp[k] = c_p * xp[:, k]
    xp[:, k + 1]=a_p * xp[:, k]+b_p * ut_p

    j0_p[k] = F * kp * ((cep) * (cs_max_p - yp[k]) * yp[k]) ** (0.5)
    k_p[k] = Jp / (2 * as_p * j0_p[k])
    eta_p[k] = R * T * np.log(k_p[k] + (k_p[k] ** 2 + 1) ** 0.5) / (F * 0.5)

    y[k] = yp[k] / (cs_max_p)  # yp is surface concentration"""
    # Molar Current Flux Density (Assumed UNIFORM for SPM)
    Jn = I[k]/Vn
    Jp = I[k]/Vp

    # Compute "current timestep" Concentration from "Battery States" via Output Eqn (Pos & Neg)
    yn[k] = C_dn @ xn[:, k] + D_dn * 0
    yp[k] = C_dp @ xp[:, k] + D_dp * 0

    # Compute "NEXT" time step "Battery States" via State Space Models (Pos & Neg)
    xn[:, [k+1]] = A_dn @ xn[:, [k]] + B_dn * Jn
    xp[:, [k+1]] = A_dp @ xp[:, [k]] + B_dp * Jp

    if k == 1:
        print(xp[:, [k+1]])

    # Compute "Exchange Current Density" per Electrode (Pos & Neg)
    i_0n = kn*F*(cen**.5) * ((yn[k])**.5) * ((cs_max_n - yn[k])**.5)
    i_0p = kp*F*(cep**.5) * ((yp[k])**.5) * ((cs_max_p - yp[k])**.5)

    # Compute Electrode "Overpotentials"
    eta_n = ((2*R*T)/F)*asinh((Jn*F)/(2*i_0n))
    eta_p = ((2*R*T)/F)*asinh((Jp*F)/(2*i_0p))

    # Record SOC of Cell
    theta_n[k] = yn[k]/cs_max_n
    theta_p[k] = yp[k]/cs_max_p

    U_n = OCV_Anode(theta_n[k])
    U_p = OCV_Cathod(theta_p[k])

    V_term[k] = U_p - U_n + eta_p - eta_n





plt.figure(1)
plt.title("Terminal Voltage vs time")
plt.xlabel("Time [sec]")
plt.ylabel("Volts")
plt.plot(time,V_term)

plt.figure(2)
plt.title("Input Current vs time")
plt.xlabel("Time [sec]")
plt.ylabel("Current")
plt.plot(time,I)

plt.figure(3)
plt.title("SOC vs time")
plt.xlabel("Time [sec]")
plt.ylabel("State of Charg")
plt.plot(time,theta_n)
plt.show()
from SPMBatteryParams import *
import numpy as np

import matplotlib.pyplot as plt
from math import asinh, tanh, cosh


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

plt.figure(0)
plt.plot(range(0, Kup), I)
plt.show()

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


"""# Sensitivity Analysis
coef = 3/(F*Rp**6 * as_p**2 * Ar_p*Lp)
Sepsi_A = np.array([[0, 1, 0], [0, 0, 1], [0, -(3465*Ds_p**2)/Rp**4,  -(189*Ds_p)/Rp**2]])
Sepsi_B = np.array([[0], [0], [1]])
Sepsi_C = coef*np.array([10395*Ds_p**2, 1260*Ds_p*Rp**2, 21*Rp**4])
Sepsi_D = np.array([0])
Sepsi_p = np.zeros([3, Kup+1])

[n, m] = np.shape(Sepsi_A)
Sepsi_A_dp = np.eye(n) + Sepsi_A*Ts
Sepsi_B_dp = Sepsi_B*Ts
Sepsi_a_p = Sepsi_A_dp
Sepsi_b_p = Sepsi_B_dp
Sepsi_c_p = Sepsi_C
Sepsi_d_p = Sepsi_D

Sepsi_p[:, [1]] = np.array([[0], [0], [0]])

# sensitivity realization in time domain for epsilon_sn from third order pade(you can refer to my slides)
coefn = 3/(F*Rn**6 * as_n**2 * Ar_n * Ln)
Sepsi_A_n = np.array([[0, 1, 0], [0, 0, 1], [0,  -(3465*Ds_n**2)/Rn**4,  -(189*Ds_n)/Rn**2]])
Sepsi_B_n = np.array([[0], [0], [1]])
Sepsi_C_n = coefn*np.array([[10395*Ds_n**2, 1260*Ds_n*Rn**2, 21*Rn**4]])
Sepsi_D_n = np.array([0])
Sepsi_n = np.zeros([3, Kup+1])


[n, m] = np.shape(Sepsi_A_n)
Sepsi_A_dn = np.eye(n) + Sepsi_A_n*Ts
Sepsi_B_dn = Sepsi_B_n*Ts
Sepsi_a_n = Sepsi_A_dn
Sepsi_b_n = Sepsi_B_dn
Sepsi_c_n = Sepsi_C_n
Sepsi_d_n = Sepsi_D_n

Sepsi_n[:, [1]] = np.array([[0], [0], [0]])

# sensitivity realization in time domain for D_sp from third order pade

coefDsp = (63*Rp)/(F*as_p*Ar_p*Lp*Rp**8)
Sdsp_A = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-(12006225*Ds_p**4)/Rp**8, -1309770*Ds_p**3/Rp**6, -42651*Ds_p**2/Rp**4, -378*Ds_p/Rp**2]])
Sdsp_B = np.array([[0], [0], [0], [1]])
Sdsp_C = coefDsp*np.array([[38115*Ds_p**2, 1980*Ds_p*Rp**2, 43*Rp**4, 0]])
Sdsp_D = np.array([0])
Sdsp_p = np.zeros([4, Kup+1])


[n, m] = np.shape(Sdsp_A)
Sdsp_A_dp = np.eye(n) + Sdsp_A*Ts
Sdsp_B_dp = Sdsp_B*Ts
Sdsp_a_p = Sdsp_A_dp
Sdsp_b_p = Sdsp_B_dp
Sdsp_c_p = Sdsp_C
Sdsp_d_p = Sdsp_D

Sdsp_p[:, [1]] = np.array([[0], [0], [0], [0]])

# sensitivity realization in time domain for D_sn from third order pade

coefDsn = (63*Rn)/(F*as_n*Ar_n*Ln*Rn**8)
Sdsn_A = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-(12006225*Ds_n**4)/Rn**8, -1309770*Ds_n**3/Rn**6, -42651*Ds_n**2/Rn**4, -378*Ds_n/Rn**2]])
Sdsn_B = np.array([[0], [0], [0], [1]])
Sdsn_C = coefDsn*np.array([[38115*Ds_n**2, 1980*Ds_n*Rn**2, 43*Rn**4,  0]])
Sdsn_D = np.array([[0]])
Sdsn_n = np.zeros([4, Kup+1])


[n, m] = np.shape(Sdsn_A)
Sdsn_A_dn = np.eye(n) + Sdsn_A*Ts
Sdsn_B_dn = Sdsn_B*Ts
Sdsn_a_n = Sdsn_A_dn
Sdsn_b_n = Sdsn_B_dn
Sdsn_c_n = Sdsn_C
Sdsn_d_n = Sdsn_D

Sdsn_n[:, [1]] = np.array([[0], [0], [0], [0]])
"""

# Initialize ALL Sensitivity Scalar Values
out_Sepsi_p = out_Sepsi_n = out_Sdsp_p = out_Sdsn_n = docvp_dCsep = docvn_dCsen = rho1p_1 = rho1p = rho2p = rho1n_1\
    = rho1n = rho2n = sen_out_spsi_p = sen_out_spsi_n = out_deta_p_desp = out_deta_n_desn = out_semi_linear_p = \
    out_semi_linear_n = sen_out_ds_p = sen_out_ds_n = dV_dDsp = dV_dDsn = dCse_dDsp = eta_p = eta_n = np.zeros(Kup)


x = np.zeros([1, Kup+1])
y = np.zeros([1, Kup+1])
k_p = np.zeros([1, Kup+1])
k_n = np.zeros([1, Kup+1])

j0_p = np.zeros([1, Kup+1])
j0_n = np.zeros([1, Kup+1])
j1 = np.zeros(Kup+1)

k_n[0] = kn
k_p[0] = kp


for k in range(0, Kup):
    time[k] = k
    j1[k] = k*Ts

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

    # if k == 1:
    #     print(xp[:, [k+1]])

    # Compute "Exchange Current Density" per Electrode (Pos & Neg)
    i_0n = j0_n[[k]] = kn*F*(cen**.5) * ((yn[k])**.5) * ((cs_max_n - yn[k])**.5)
    i_0p = j0_p[[k]] = kp*F*(cep**.5) * ((yp[k])**.5) * ((cs_max_p - yp[k])**.5)

    # IDK YET Conductivity Value for Electrolyte????
    if k == 0:
        break
    else:
        k_n[k] = Jn / (2 * as_n * j0_n[[k]])
        k_p[k] = Jp / (2 * as_p * j0_p[[k]])

    # Compute Electrode "Overpotentials"
    eta_n[k] = R*T*np.log(k_n[k]+(k_n[k]**2+1)**0.5)/(F*0.5)
    eta_p[k] = R*T*np.log(k_p[k]+(k_p[k]**2+1)**0.5)/(F*0.5)

    # eta_n = ((2*R*T)/F)*asinh((Jn*F)/(2*i_0n))
    # eta_p = ((2*R*T)/F)*asinh((Jp*F)/(2*i_0p))

    # Record SOC of Cell
    theta_n[k] = x[k] = yn[k]/cs_max_n
    theta_p[k] = y[k] = yp[k]/cs_max_p

    U_n = OCV_Anode(theta_n[k])
    U_p = OCV_Cathod(theta_p[k])

    V_term[k] = U_p - U_n + eta_p - eta_n

    """# state space realization for epsilon_sp
    out_Sepsi_p[k] = Sepsi_c_p@Sepsi_p[:, [k]]
    Sepsi_p[:, [k+1]] = Sepsi_a_p@Sepsi_p[:, [k]] + Sepsi_b_p*I[k]
    # current input for positive electrode is negative, therefore the sensitivity output should be multiplied by -1
    # state space realization for epsilon_sn

    out_Sepsi_n[k] = Sepsi_c_n@Sepsi_n[:, [k]]
    Sepsi_n[:, [k+1]] = Sepsi_a_n@Sepsi_n[:, [k]] + Sepsi_b_n*I[k]

    # state space realization for D_sp
    out_Sdsp_p[k] = Sdsp_c_p@Sdsp_p[:, [k]]
    Sdsp_p[:, [k+1]] = Sdsp_a_p@Sdsp_p[:, [k]] + Sdsp_b_p*I[k]

    # state space realization for D_sn
    out_Sdsn_n[k] = Sdsn_c_n@Sdsn_n[:, [k]]
    Sdsn_n[:, [k+1]] = Sdsn_a_n@Sdsn_n[:, [k]] + Sdsn_b_n*I[k]

    # OcP slope
    docvp_dCsep[k] = 0.07645*(-54.4806/cs_max_p)*((1.0/cosh(30.834-54.4806*y[0, [k]]))**2)
    + 2.1581*(-50.294/cs_max_p)*((cosh(52.294-50.294*y[0, [k]]))**(-2))
    + 0.14169*(19.854/cs_max_p)*((cosh(11.0923-19.8543*y[0, [k]]))**(-2))
    - 0.2051*(5.4888/cs_max_p)*((cosh(1.4684-5.4888*y[0, [k]]))**(-2))
    - 0.2531/0.1316/cs_max_p*((cosh((-y[0, [k]]+0.56478)/0.1316))**(-2))
    - 0.02167/0.006/cs_max_p*((cosh((y[0, [k]]-0.525)/0.006))**(-2))

    docvn_dCsen[k] = -1.5*(120.0/cs_max_n)*np.exp(-120.0*x[0, [k]])
    + (0.0351/(0.083*cs_max_n))*((cosh((x[0, [k]]-0.286)/0.083))**(-2))
    - (0.0045/(cs_max_n*0.119))*((cosh((x[0, [k]]-0.849)/0.119))**(-2))
    - (0.035/(cs_max_n*0.05))*((cosh((x[0, [k]]-0.9233)/0.05))**(-2))
    - (0.0147/(cs_max_n*0.034))*((cosh((x[0, [k]]-0.5)/0.034))**(-2))
    - (0.102/(cs_max_n*0.142))*((cosh((x[0, [k]]-0.194)/0.142))**(-2))
    - (0.022/(cs_max_n*0.0164))*((cosh((x[0, [k]]-0.9)/0.0164))**(-2))
    - (0.011/(cs_max_n*0.0226))*((cosh((x[0, [k]]-0.124)/0.0226))**(-2))
    + (0.0155/(cs_max_n*0.029))*((cosh((x[0, [k]]-0.105)/0.029))**(-2))

    #
    rho1p_1[k] = -np.sign(I[k])*(-3*R*T)/(0.5*F*Rp*as_p) * ((1+1/(k_p[k])**2)**(-0.5))
    rho1p[k] = R*T/(0.5*F) * (1/(k_p[k]+(k_p[k]**2+1)**0.5)) * (1+k_p[k]/((k_p[k]**2+1)**0.5)) * (-3*Jp/(2*as_p**2*j0_p[k]*Rp))
    rho2p[k] = (R*T)/(2*0.5*F) * (cep*cs_max_p-2*cep*yp[k])/(cep*yp[k]*(cs_max_p-yp[k])) * (1+1/(k_p[k])**2)**(-0.5)

    rho1n_1[k] = np.sign(I[k])*(-3*R*T)/(0.5*F*Rn*as_n) * ((1+1/(k_n[k])**2)**(-0.5))
    rho1n[k] = R*T/(0.5*F) * (1/(k_n[k]+(k_n[k]**2+1)**0.5)) * (1+k_n[k]/((k_n[k]**2+1)**0.5)) * (-3*Jn/(2*as_n**2*j0_n[k]*Rn))
    rho2n[k] = (-R*T)/(2*0.5*F) * (cen*cs_max_n-2*cen*yn[k])/(cen*yn[k]*(cs_max_n-yn[k])) * (1+1/(k_n[k])**2)**(-0.5)

    # sensitivity of epsilon_sp    epsilon_sn
    sen_out_spsi_p[k] = (rho1p[k]+(rho2p[k] + docvp_dCsep[k])*-out_Sepsi_p[k])
    sen_out_spsi_n[k] = (rho1n[k]+(rho2n[k]+docvn_dCsen[k])*out_Sepsi_n[k])

    out_deta_p_desp[k] = rho1p[k] + (rho2p[k])*-out_Sepsi_p[k]
    out_deta_n_desn[k] = rho1n[k] + (rho2n[k])*out_Sepsi_n[k]

    out_semi_linear_p[k] = (docvp_dCsep[k])*out_Sepsi_p[k]
    out_semi_linear_n[k] = (docvn_dCsen[k])*out_Sepsi_n[k]

    # Sensitivity of Dsp Dsn
    sen_out_ds_p[k] = ((rho2p[k] + docvp_dCsep[k])*-out_Sdsp_p[k])*Ds_p
    sen_out_ds_n[k] = ((rho2n[k] + docvn_dCsen[k])*out_Sdsn_n[k])*Ds_n

    dV_dDsp[k] = sen_out_ds_p[k]
    dV_dDsn[k] = sen_out_ds_n[k]

    dCse_dDsp[k] = -out_Sdsp_p[k]*Ds_p
"""


plt.figure(1)
plt.title("Terminal Voltage vs time")
plt.xlabel("Time [sec]")
plt.ylabel("Volts")
plt.plot(time, V_term)

plt.figure(2)
plt.title("Input Current vs time")
plt.xlabel("Time [sec]")
plt.ylabel("Current")
plt.plot(time, I)

plt.figure(3)
plt.title("SOC vs time")
plt.xlabel("Time [sec]")
plt.ylabel("State of Charge")
plt.plot(time, theta_n)

# plt.figure(4)
# plt.title("Sensitivity vs time")
# plt.xlabel("Time [sec]")
# plt.ylabel("Sensitivity")
# plt.plot(time, sen_out_spsi_p*epsilon_sp)
plt.show()
# # from SPMeBatteryParams import *
# from math import asinh, tanh, cosh
# import numpy as np
from SPMe_Baseline_Params import SPMe_Baseline_Parameters

class dummy(SPMe_Baseline_Parameters):
    def __init__(self):

        self.param = {}

        self.param_key_list = ['epsilon_sn', 'epsilon_sp', 'epsilon_e_n', 'epsilon_e_p',
                               'F', 'Rn', 'Rp', 'R', 'T', 'Ar_n', 'Ar_p', 'Ln', 'Lp', 'Lsep', 'Lc',
                               'Ds_n', 'Ds_p', 'De', 'De_p', 'De_n', 'kn', 'kp', 'stoi_n0', 'stoi_n100',
                               'stoi_p0', 'stoi_p100', 'SOC', 'cs_max_n', 'cs_max_p', 'Rf', 'as_n', 'as_p',
                               'Vn', 'Vp', 't_plus', 'cep', 'cen', 'rfa_n', 'rfa_p', 'epsi_sep', 'epsi_e',
                               'epsi_n', 'gamak', 'kappa', 'kappa_eff', 'kappa_eff_sep']

        self.default_param_vals = [self.epsilon_sn, self.epsilon_sp, self.epsilon_e_n, self.epsilon_e_p,
                                   self.F, self.Rn, self.Rp, self.R, self.T, self.Ar_n, self.Ar_p, self.Ln,
                                   self.Lp, self.Lsep, self.Lc, self.Ds_n, self.Ds_p, self.De, self.De_p,
                                   self.De_n, self.kn, self.kp, self.stoi_n0, self.stoi_n100, self.stoi_p0,
                                   self.stoi_p100, self.SOC, self.cs_max_n, self.cs_max_p, self.Rf, self.as_n,
                                   self.as_p, self.Vn, self.Vp, self.t_plus, self.cep, self.cen, self.rfa_n,
                                   self.rfa_p, self.epsi_sep, self.epsi_e, self.epsi_n, self.gamak, self.kappa,
                                   self.kappa_eff, self.kappa_eff_sep]

        self.param = {self.param_key_list[i]: self.default_param_vals[i] for i in range(0, len(self.param_key_list))}

        print(self.expand_parameters())


    def expand_parameters(self):
        empty_list = []
        param_key_list = list(self.param)
        print(len(self.param))

        empty_list = [self.param[param_key_list[i]] for i in range(len(self.param))]

        return empty_list

thing = dummy()
"""new_diction = {}

new_diction['key1'] = 5

new_diction['key2'] = float(10)


print(new_diction)"""
# theta_p = .5
# k_p = .5
# cep = 1000
# cs_max_p = 5.1219e+04
# Jp = .5
# j0_p = .5
# out_Sepsi_p = .5
# docvp_dCsep = .5
#
#
#
# rho1p = R * T / (0.5 * F) * (1 / (k_p + (k_p ** 2 + 1) ** 0.5)) * (1 + k_p / ((k_p ** 2 + 1) ** 0.5)) * (
#                 -3 * Jp / (2 * as_p ** 2 * j0_p * Rp))
#
# rho2p = (R * T) / (2 * 0.5 * F) * (cep * cs_max_p - 2 * cep * theta_p) / (
#                     cep * theta_p * (cs_max_p - theta_p)) * (1 + 1 / (k_p) ** 2) ** (-0.5)
#
# sen_out_spsi_p = (rho1p + (rho2p + docvp_dCsep) * -out_Sepsi_p)
#
# print(rho1p)
# print(rho2p)
# print(sen_out_spsi_p)
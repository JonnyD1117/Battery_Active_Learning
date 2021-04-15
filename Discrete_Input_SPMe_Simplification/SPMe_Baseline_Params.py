class SPMe_Baseline_Parameters:
    # Scott's parameters
    """
    This is the file which stores Battery Parameters for Scott Moura's Single Particle Battery Model (Provided by Qingzhi)
    """


    epsilon_sn = 0.6  # average negative active volume fraction
    epsilon_sp = 0.50  # average positive active volume fraction
    epsilon_e_n = 0.3  # Liquid [electrolyte] volume fraction (pos & neg)
    epsilon_e_p = 0.3

    # F = 96485.3329      # Faraday constant
    F = 96487  # Faraday constant
    Rn = 10e-6  # Active particle radius (pose & neg)
    Rp = 10e-6

    R = 8.314  # Universal gas constant
    T = 298.15  # Ambient Temp. (kelvin)

    Ar_n = 1  # Current collector area (anode & cathode)
    Ar_p = 1

    Ln = 100e-6  # Electrode thickness (pos & neg)
    Lp = 100e-6
    Lsep = 25e-6  # Separator Thickness
    Lc = Ln + Lp + Lsep  # Total Cell Thickness

    Ds_n = 3.9e-14  # Solid phase diffusion coefficient (pos & neg)
    Ds_p = 1e-13
    De = 2.7877e-10  # Electrolyte Diffusion Coefficient
    De_p = De
    De_n = De

    kn = 1e-5 / F  # Rate constant of exchange current density (Reaction Rate) [Pos & neg]
    kp = 3e-7 / F

    # Stoichiometric Coef. used for "interpolating SOC value based on OCV Calcs. at 0.0069% and 0.8228%
    stoi_n0 = 0.0069  # Stoich. Coef. for Negative Electrode
    stoi_n100 = 0.6760

    stoi_p0 = 0.8228  # Stoich. Coef for Positive Electrode
    stoi_p100 = 0.442

    SOC = 1  # SOC can change from 0 to 1

    # # Interpolate Value of SOC (aka - stoich. coef.) given min & max coef. values
    # stoi_x = (stoi_x100-stoi_x0)*SOC+stoi_x0      # Positive Electrode Interpolant
    # stoi_y = stoi_y0-(stoi_y0-stoi_y100)*SOC      # Negative Electrode Interpolant

    cs_max_n = (3.6e3 * 372 * 1800) / F  # 0.035~0.870=1.0690e+03~ 2.6572e+04
    cs_max_p = (3.6e3 * 274 * 5010) / F  # Positive electrode  maximum solid-phase concentration 0.872~0.278=  4.3182e+04~1.3767e+04

    Rf = 1e-3  #
    as_n = 3 * epsilon_sn / Rn  # Active surface area per electrode unit volume (Pos & Neg)
    as_p = 3 * epsilon_sp / Rp

    Vn = Ar_n * Ln  # Electrode volume (Pos & Neg)
    Vp = Ar_p * Lp

    t_plus = 0.4

    cep = 1000  # Electrolyte Concentration (Assumed Constant?) [Pos & Neg]
    cen = 1000

    # Common Multiplicative Factor use in SS  (Pos & Neg electrodes)
    rfa_n = 1 / (F * as_n * Rn ** 5)
    rfa_p = 1 / (F * as_p * Rp ** 5)

    epsi_sep = 1
    epsi_e = 0.3
    epsi_n = epsi_e
    gamak = (1 - t_plus) / (F * Ar_n)

    kappa = 1.1046
    kappa_eff = kappa * (epsi_e ** 1.5)
    kappa_eff_sep = kappa * (epsi_sep ** 1.5)



if __name__ == "__main__":


    obj = SPMe_Baseline_Parameters()

    print('cs_max_n', obj.cs_max_n)
    print('cs_max_p', obj.cs_max_p)



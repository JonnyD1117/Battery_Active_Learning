import scipy.io
import numpy as np
import matplotlib.pyplot as plt


FUDs_direct_model_path = "C:\\Users\\Indy-Windows\\Pictures\\Lin\\Logger_Comparison\\DirectModel\\FUDs\\mfile\\Direct_FUDs_Mat_File"
FUDS_env_model_path = "C:\\Users\\Indy-Windows\\Pictures\\Lin\\Logger_Comparison\\EnvModel\\FUDs\\mfiles\\Env_FUDs_Mat_File.mat"

Pulse_direct_model_path = "C:\\Users\\Indy-Windows\\Pictures\\Lin\\Logger_Comparison\\DirectModel\\Pulse\\mfile\\Direct_Pulses_Mat_File"
Pulse_env_model_path = "C:\\Users\\Indy-Windows\\Pictures\\Lin\\Logger_Comparison\\EnvModel\\Pulses\\mfile\\Env_Pulses_Mat_File.mat"

Zero_CC_direct_model_path = "C:\\Users\\Indy-Windows\\Pictures\\Lin\\Logger_Comparison\\DirectModel\\Zero_CC\\mfile\\Direct_Zero_CC_Mat_File"
Zero_CC_env_model_path = "C:\\Users\\Indy-Windows\\Pictures\\Lin\\Logger_Comparison\\EnvModel\\Zero_CC\\mfile\\Env_Zero_CC_Mat_File.mat"


fud_direct = scipy.io.loadmat(FUDs_direct_model_path)
fud_env = scipy.io.loadmat(FUDS_env_model_path)

pulse_direct = scipy.io.loadmat(Pulse_direct_model_path)
pulse_env = scipy.io.loadmat(Pulse_env_model_path)

zeroCC_direct = scipy.io.loadmat(Zero_CC_direct_model_path)
zeroCC_env = scipy.io.loadmat(Zero_CC_env_model_path)



# mfile_data = {'input_current':current, 'SOC': soc, 'time':time, 'V_term':V_term, 'Epsi_sp':dV_dEpsi_sp}

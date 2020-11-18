from SPMe_w_Sensitivity_Params import SingleParticleModelElectrolyte_w_Sensitivity
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import scipy.io
import numpy as np


# SPMe_Model = SingleParticleModelElectrolyte_w_Sensitivity(timestep=.2)

# Import FUDs profile
# mat = scipy.io.loadmat("I_FUDS.mat")
mat = scipy.io.loadmat("Test_Files/Test_Data_mfiles/Env_Zero_CC_Mat_File.mat")

# 'V_term':V_term, 'Epsi_sp'

I_fuds = mat['input_current'][0][:]
time_fuds = mat["time"][0][:]

plt.plot(I_fuds)
plt.show()




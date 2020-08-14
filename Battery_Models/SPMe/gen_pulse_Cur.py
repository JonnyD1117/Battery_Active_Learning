import numpy as np
import scipy.io
from scipy import signal
from matplotlib import pyplot as plt

# sqwave = sign(sin(2*pi*f*t)) #an actual square wave



t = np.arange(0,1000,.1)
# linspace(0, 1000 , 1000, endpoint=False)

sig = signal.square(2 * np.pi * (1/60) * t)
mag_sig = 25.67*sig

# plt.plot(t, mag_sig)
# plt.show()

scipy.io.savemat("correct_pulse_input.mat", {"I": mag_sig, "time": t})



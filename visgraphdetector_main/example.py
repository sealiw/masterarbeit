#import packages
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import electrocardiogram
from visgraphdetector import VisGraphDetector
#%%
ecg = electrocardiogram()
fs = 360
beta = 0.55
gamma = 0.5
lowcut = 4
M = 2*fs
time = np.arange(0,len(ecg)/360,1/360)

vgd = VisGraphDetector(360)
R_peaks, weights, weighted_signal = vgd.visgraphdetect(ecg, beta=beta, gamma=gamma, lowcut=lowcut, M = M)

plt.plot(time,ecg)
plt.plot(time[R_peaks],ecg[R_peaks],'X')

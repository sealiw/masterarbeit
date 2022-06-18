import pywt
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from stage1 import *
from helper_functions import *
from ecgdetectors import Detectors
import padasip as pa




displacement, tfm_ecg2, t_radar, fs_radar, t_ecg, fs_ecg = data_acquisition()
delay = 100
dsplm = np.pad(displacement, (delay, 0), 'edge')

N = len(displacement)
log_d = np.zeros(N)
log_y = np.zeros(N)
y = np.zeros(N)
for k in range(N):
    y[k] = (dsplm[k+delay] - dsplm[k])**2

plt.figure(figsize=(15,9))
plt.subplot(211);plt.title("Adaptation");plt.xlabel("samples - k")
plt.plot(displacement,"b", label="d - target")
plt.plot(y,"g", label="y - output");plt.legend()
plt.subplot(212);plt.title("Filter error");plt.xlabel("samples - k")
plt.plot(10*np.log10(y-displacement),"r", label="e - error [dB]")
plt.legend(); plt.tight_layout(); plt.show()

# filter_length = 9000
# x = np.pad(displacement, (int(filter_length / 2), int((filter_length - 1) / 2)), 'edge')
# sig_subtr = np.convolve(x, np.ones(filter_length) / filter_length, 'valid')
# plt.specgram(sig_subtr, int(fs_radar * 60), fs_radar, noverlap=int(fs_radar * 60*0.5), scale='dB', detrend='mean')
# plt.ylim((0, 5))
# plt.colorbar()

# radar_peaks = find_radar_peaks(fs_radar, sig_subtr)
# peak_signal = displacement[radar_peaks]
#
# plt.figure()
# plt.plot(t_radar, sig_subtr)
# plt.plot(t_radar[radar_peaks], sig_subtr[radar_peaks], 'X')

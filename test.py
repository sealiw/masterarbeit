import pywt
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from stage1 import *
from helper_functions import *

displacement, tfm_ecg2, t_radar, fs_radar, t_ecg, fs_ecg = data_acquisition()

filter_length = 250
x = np.pad(displacement, (int(filter_length / 2), int((filter_length - 1) / 2)), 'edge')
sig_subtr = displacement - np.convolve(x, np.ones(filter_length) / filter_length, 'valid')

# sig_subtr = displacement - running_robust_location(displacement, n=filter_length, method='median')

# filter_displacement1 = signal.butter(10, 18, 'lowpass', fs=fs_radar, output='sos')
# sig_subtr = displacement - signal.sosfilt(filter_displacement1, displacement)


filter_displacement2 = signal.butter(10, 20, 'lowpass', fs=fs_radar, output='sos')
sig_fil = signal.sosfilt(filter_displacement2, sig_subtr)

haar = pywt.Wavelet('haar')
# cA= pywt.swt(displacement, haar, 5)

wvlt_displacement = pywt.swt(sig_fil, haar, level=2, start_level=3)[0][1]

ran = np.arange(100000, 200000)

peaks_radar, _, _ = find_ecg_peaks(fs_radar, sig_fil[ran], negative=False)
peaks_wvlt, _, _ = find_ecg_peaks(fs_radar, wvlt_displacement[ran], negative=False)
plt.figure()
plt.plot(sig_fil[ran])
plt.plot(wvlt_displacement[ran])
plt.plot(peaks_radar, sig_fil[ran][peaks_radar], 'X')
plt.plot(peaks_wvlt, wvlt_displacement[ran][peaks_wvlt], 'X')

# nums = len(wvlt_displacement)
# fig, axes = plt.subplots(nums, 1)
# peaks_radar, _, _ = find_ecg_peaks(fs_radar, sig_fil[ran], negative=False)
# for index, wvlt in enumerate(wvlt_displacement):
#     peaks_wvlt, _, _ = find_ecg_peaks(fs_radar, wvlt[1][ran], negative=False)
#     axes[index].plot(sig_fil[ran])
#     axes[index].plot(wvlt[1][ran])
#     axes[index].plot(peaks_radar, sig_fil[ran][peaks_radar], 'X')
#     axes[index].plot(peaks_wvlt, wvlt[1][ran][peaks_wvlt], 'X')
#     # axes[index].plot(tfm_ecg2[ran]/1000)

print('done')

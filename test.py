import pywt
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from stage1 import *
from helper_functions import *

displacement, tfm_ecg2, t_radar, fs_radar, t_ecg, fs_ecg = data_acquisition()

filter_length = 10
x = np.pad(displacement, (int(filter_length / 2), int((filter_length - 1) / 2)), 'edge')
sig_subtr = displacement - np.convolve(x, np.ones(filter_length) / filter_length, 'valid')

filter_displacement = signal.butter(10, 20, 'lowpass', fs=fs_radar, output='sos')
sig_fil = signal.sosfilt(filter_displacement, sig_subtr)

haar = pywt.Wavelet('db1')
# cA= pywt.swt(displacement, haar, 5)

wvlt_displacement = pywt.swt(sig_fil, haar, 5)

ran = np.arange(90000, 120000)
nums = len(wvlt_displacement)
fig, axes = plt.subplots(nums, 1)
for index, wvlt in enumerate(wvlt_displacement):
    peaks_radar, _, _ = find_ecg_peaks(fs_radar, wvlt[1][ran], negative=False)
    axes[index].plot(sig_fil[ran])
    axes[index].plot(wvlt[1][ran])
    axes[index].plot(peaks_radar, wvlt[1][ran][peaks_radar], 'X')
    # axes[index].plot(tfm_ecg2[ran]/1000)

print('done')

import numpy as np

import matplotlib.pyplot as plt
import pywt
from stage1 import *
from helper_functions import *

displacement, tfm_ecg2, tfm_bp, t_radar, fs_radar, t_ecg, fs_ecg, t_bp, fs_bp = data_acquisition()

pulse_wave_swt, heart_sounds = get_useful_signals(displacement, fs_radar)

peaks_ecg = find_peaks(tfm_ecg2, t_ecg, 160, 4)
HR_ecg = compute_heart_rate(peaks_ecg, fs_ecg)

# Find peaks from pulse-wave
peaks_pulse = find_peaks(pulse_wave_swt, t_radar, 1200, 1, [1200, 0.7e-6])
# Estimate heart-rate from pulse-wave
HR_pulse = compute_heart_rate(peaks_pulse, fs_radar)
HR = running_robust_location(HR_pulse, method='m_loc_hub', n=3, c=0.01)

# Find peaks from heart_sound
peaks_sound = find_peaks(heart_sounds, t_radar, 400, 2, [2000, 0.1e-6])


PATs, valid_peaks = compute_PAT(peaks_pulse, peaks_sound, fs_radar)

SBP_valid, DBP_valid = get_BP_values(tfm_bp, fs_bp, fs_radar, valid_peaks)

HR_valid = HR[np.isin(peaks_pulse[:-1], valid_peaks)]

beta = linear_regression(SBP_valid, DBP_valid, PATs, HR_valid)




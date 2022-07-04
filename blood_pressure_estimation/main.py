import numpy as np

import matplotlib.pyplot as plt
import pywt
from stage1 import *
from helper_functions import *

displacement, tfm_ecg2, tfm_bp, t_radar, fs_radar, t_ecg, fs_ecg, t_bp, fs_bp = data_acquisition()

pulse_wave_swt, heart_sounds = get_useful_signals(displacement, fs_radar)
pulse_wave_swt = pulse_wave_swt[:-1]
heart_sounds = heart_sounds[:-1]
t_radar = t_radar[:-1]
tfm_bp = tfm_bp[:-1]
t_bp = t_bp[:-1]
t_ecg = t_ecg[:-1]
tfm_ecg2 = tfm_ecg2[:-1]

peaks_ecg = find_peaks(tfm_ecg2, t_ecg, 160, 4)
# peaks_ecg = find_ecg_peaks(fs_ecg, tfm_ecg2)
IBI_ecg = compute_IBI(peaks_ecg, fs_ecg)

# Find peaks from pulse-wave
peaks_pulse = find_peaks(pulse_wave_swt, t_radar, 1200, 1, [1200, 0.1e-6])
valleys_pulse, peaks_interm = find_valleys_and_interm_peaks(pulse_wave_swt, t_radar, peaks_pulse)
plt.figure()
plt.plot(t_radar, pulse_wave_swt)
plt.plot(t_radar[peaks_pulse], pulse_wave_swt[peaks_pulse], 'X')
plt.plot(t_radar[valleys_pulse], pulse_wave_swt[valleys_pulse], 'X')
plt.plot(t_radar[peaks_interm], pulse_wave_swt[peaks_interm], 'X')


# Estimate heart-rate from pulse-wave
IBI_pulse = compute_IBI(peaks_pulse, fs_radar)
IBI_pulse_final = running_robust_location(IBI_pulse, method='m_loc_hub', n=1, c=0.01)
HR = 60/running_robust_location(IBI_pulse, method='m_loc_hub', n=5, c=2)

plt.figure()
# plt.plot(t_ecg[R_peaks_ecg[:-1]], HR_ref)
plt.plot(t_ecg[peaks_ecg[:-1]], IBI_ecg)
plt.plot(t_radar[peaks_pulse[:-1]], IBI_pulse_final)

# Find peaks from heart_sound
peaks_sound = find_peaks(heart_sounds, t_radar, 400, 2, [2000, 0.1e-6])
plt.figure()
plt.plot(t_radar, heart_sounds)
plt.plot(t_radar[peaks_sound], heart_sounds[peaks_sound], 'X')


PATs_peaks, valid_peaks = compute_PAT(peaks_pulse, peaks_sound, fs_radar)
peak_heights = pulse_wave_swt[valid_peaks]
valid_valleys, valid_interm_peaks = get_valid_valleys_and_interm_peaks(valleys_pulse, peaks_interm, valid_peaks)
PATs_valleys = compute_PAT_valleys(valid_valleys, peaks_sound, fs_radar)
peak_2_valley_time = (valid_peaks - valid_valleys)/fs_radar
peak_2_interm_peak_time = (valid_peaks - valid_interm_peaks)/fs_radar

valley_heights = pulse_wave_swt[valid_valleys]
interm_peak_heights = pulse_wave_swt[valid_interm_peaks]


SBP_valid, DBP_valid = get_BP_values(tfm_bp, t_bp, fs_bp, fs_radar, valid_peaks)

HR_valid = HR[np.isin(peaks_pulse[:-1], valid_peaks)]

ratio = 0.5
learn_inds = np.arange(0, ratio * len(PATs_peaks), 1, dtype=int)
pred_inds = np.arange(ratio * len(PATs_peaks), len(PATs_peaks), 1, dtype=int)


A1 = np.stack((np.log(PATs_peaks), np.log(PATs_valleys), HR_valid, peak_heights, valley_heights, interm_peak_heights, peak_2_valley_time, peak_2_interm_peak_time), axis=1)
A2 = np.stack((np.log(PATs_peaks), HR_valid, np.ones(len(PATs_peaks))), axis=1)

beta1, beta2 = linear_regression(SBP_valid[learn_inds], DBP_valid[learn_inds], A1[learn_inds], A2[learn_inds])



SBP_pred1 = A1@beta1
SBP_pred2 = A2@beta2

plt.figure()
plt.plot(SBP_valid, label='SBP')
plt.plot(SBP_pred1, label='Predicted SBP')
plt.plot(SBP_pred2, label='Predicted SBP2')
# plt.plot(PATs, label='PATs')
# plt.plot(HR_valid, label='Heart rate')
plt.axvline(learn_inds[-1], color='r')
plt.legend()


for feature in [np.log(PATs_peaks), np.log(PATs_valleys), HR_valid, peak_heights, valley_heights, interm_peak_heights, peak_2_valley_time, peak_2_interm_peak_time]:
    print(np.corrcoef(SBP_valid[pred_inds], feature[pred_inds])[0, 1])


print('Mean Error with peak-heights: ', np.mean(abs(SBP_valid-SBP_pred1.squeeze())[pred_inds]))
print('1.96xSTD with peak-heights: ', 1.96*np.std(abs(SBP_valid-SBP_pred1.squeeze())[pred_inds]))
print('Mean Error without peak-heights: ', np.mean(abs(SBP_valid-SBP_pred2.squeeze())[pred_inds]))
print('1.96xSTD without peak-heights: ', 1.96*np.std(abs(SBP_valid-SBP_pred2.squeeze())[pred_inds]))

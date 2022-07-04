import matplotlib.pyplot as plt

from stage1 import *
from scipy import signal
from helper_functions import *

displacement, tfm_ecg2, tfm_bp, t_radar, fs_radar, t_ecg, fs_ecg, t_bp, fs_bp = data_acquisition()

# sos_heart_sounds = signal.butter(4, [20, 40], 'bandpass', fs=fs_radar, output='sos')
# heart_sounds = signal.sosfiltfilt(sos_heart_sounds, displacement)
#
# peaks = find_peaks(heart_sounds, t_radar, 200, 1)
# f = interpolate.interp1d(t_radar[peaks], heart_sounds[peaks], kind='linear')
# t_new = t_radar[peaks[0]:peaks[-1]+1]
# peak_signal = f(t_new)
# plt.figure()
# plt.plot(t_radar, heart_sounds)
# plt.plot(t_radar[peaks], heart_sounds[peaks])
# heart_sound_peaks = find_ecg_peaks(fs_radar, peak_signal)
# heart_sound_peaks = heart_sound_peaks + np.where(t_radar == t_new[0])[0][0]
# plt.plot(t_radar[heart_sound_peaks], heart_sounds[heart_sound_peaks], 'X')

# ecg_peaks0, _ = signal.find_peaks(tfm_ecg2)
# f = interpolate.interp1d(t_ecg[ecg_peaks0], tfm_ecg2[ecg_peaks0], kind='linear')
# x_new0 = t_ecg[ecg_peaks0[0]:ecg_peaks0[-1]]
# ecg_env0 = f(x_new0)
#
# ecg_peaks1, _ = signal.find_peaks(ecg_env0)
# f = interpolate.interp1d(x_new0[ecg_peaks1], ecg_env0[ecg_peaks1], kind='linear')
# x_new1 = x_new0[ecg_peaks1[0]:ecg_peaks1[-1]]
# ecg_env1 = f(x_new1)
#
# ecg_peaks2, _ = signal.find_peaks(ecg_env1)
# f = interpolate.interp1d(x_new1[ecg_peaks2], ecg_env1[ecg_peaks2], kind='linear')
# x_new2 = x_new1[ecg_peaks2[0]:ecg_peaks2[-1]]
# ecg_env2 = f(x_new2)
#
# ecg_peaks3, _ = signal.find_peaks(ecg_env2)
# f = interpolate.interp1d(x_new2[ecg_peaks3], ecg_env2[ecg_peaks3], kind='linear')
# x_new3 = x_new2[ecg_peaks3[0]:ecg_peaks3[-1]]
# ecg_env3 = f(x_new3)
#
# ecg_peaks4, _ = signal.find_peaks(ecg_env3)
# f = interpolate.interp1d(x_new3[ecg_peaks4], ecg_env3[ecg_peaks4], kind='linear')
# x_new4 = x_new3[ecg_peaks4[0]:ecg_peaks4[-1]]
# ecg_env4 = f(x_new4)

# SBP = find_ecg_peaks(fs_bp, tfm_bp)
# DBP = find_ecg_peaks(fs_bp, tfm_bp, negative=True)

SBP = find_peaks(tfm_bp, t_bp, 100, 1)
DBP = find_peaks(-tfm_bp, t_bp, 100, 1)

plt.figure()
plt.plot(t_bp, tfm_bp)
plt.plot(t_bp[SBP], tfm_bp[SBP], 'X')
plt.plot(t_bp[DBP], tfm_bp[DBP], 'X')

# plt.plot(t_ecg[ecg_peaks_ref], tfm_ecg2[ecg_peaks_ref], 'X')
# plt.plot(t_ecg[ecg_peaks], tfm_ecg2[ecg_peaks], 'X')

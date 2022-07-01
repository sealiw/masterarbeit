import numpy as np

import matplotlib.pyplot as plt
import pywt
from stage1 import *
from helper_functions import *

displacement, tfm_ecg2, tfm_bp, t_radar, fs_radar, t_ecg, fs_ecg, t_bp, fs_bp = data_acquisition()

# Filter Pulse-Wave and Heart-Sound
sos_pulse_wave = signal.butter(4, [0.75, 15], 'bandpass', fs=fs_radar, output='sos')
pulse_wave = signal.sosfiltfilt(sos_pulse_wave, displacement)

sos_heart_sounds = signal.butter(4, [20, 40], 'bandpass', fs=fs_radar, output='sos')
heart_sounds = signal.sosfiltfilt(sos_heart_sounds, displacement)

# Perform Wavelet Transform
haar = pywt.Wavelet('haar')
pulse_wave_swt = pywt.swt(pulse_wave, haar, level=1)[0][1]

heart_sounds_swt = pywt.swt(heart_sounds, haar, level=1)[0][1]

plt.figure()
plt.plot(t_radar[:-1], pulse_wave_swt[:-1])
# plt.plot(t_radar, heart_sounds/max(abs(heart_sounds)))

# Estimate Heart-rate from Pulse-Wave
peaks_pulse = find_peaks(pulse_wave_swt, t_radar, 1200, 1, [1200, 0.7e-6])
plt.plot(t_radar[peaks_pulse], pulse_wave_swt[peaks_pulse], 'X')

HR_pulse = compute_heart_rate(peaks_pulse, fs_radar)

peaks_ecg = find_peaks(tfm_ecg2, t_ecg, 160, 4)
HR_ecg = compute_heart_rate(peaks_ecg, fs_ecg)
plt.figure()
plt.plot(t_ecg, tfm_ecg2)
plt.plot(t_ecg[peaks_ecg], tfm_ecg2[peaks_ecg])

HR_pulse_final = running_robust_location(HR_pulse, method='m_loc_hub', n=3, c=0.01)

peaks_sound = find_peaks(heart_sounds, t_radar, 400, 2, [2000, 0.1e-6])
HR_sound = compute_heart_rate(peaks_sound, fs_radar)
HR_sound_final = running_robust_location(HR_sound, method='m_loc_hub', n=3, c=2)


PATs, valid_peaks = compute_PAT(peaks_pulse, peaks_sound, fs_radar)
plt.figure()
plt.plot(PATs)

SBP_inds = np.array(find_ecg_peaks(fs_bp, tfm_bp))
DBP_inds = np.array(find_ecg_peaks(fs_bp, tfm_bp, negative=True))

SBP_inds_valid = np.array([], dtype=int)
DBP_inds_valid = np.array([], dtype=int)
factor = fs_radar/fs_bp
for peak in valid_peaks:
    SBP_diff = abs(SBP_inds*factor - peak)
    closest_SBP = SBP_inds[np.where(min(SBP_diff) == SBP_diff)[0][0]]
    SBP_inds_valid = np.append(SBP_inds_valid, closest_SBP)

    DBP_diff = abs(DBP_inds*factor - peak)
    closest_DBP = DBP_inds[np.where(min(DBP_diff) == DBP_diff)[0][0]]
    DBP_inds_valid = np.append(DBP_inds_valid, closest_DBP)


plt.figure()
plt.plot(t_radar[peaks_pulse], np.zeros(len(peaks_pulse)), 'X')
plt.plot(t_radar[peaks_sound], np.zeros(len(peaks_sound)), 'X')
plt.plot(t_bp[SBP_inds_valid], np.zeros((len(SBP_inds_valid))), 'X')
plt.plot(t_bp[DBP_inds_valid], np.zeros((len(DBP_inds_valid))), 'Xr')


# evaluation(t_radar, t_ecg, fs_ecg, peaks_sound, peaks_ecg, HR_sound_final, HR_ecg)
#
plt.figure()
# plt.plot(t_ecg[R_peaks_ecg[:-1]], HR_ref)
plt.plot(t_ecg[peaks_ecg[:-1]], HR_ecg)
plt.plot(t_radar[peaks_pulse[:-1]], HR_pulse_final)
plt.plot(t_radar[peaks_sound[:-1]], HR_sound_final)
#
#
# plt.figure()
# plt.plot(t_radar, heart_sounds)
# plt.plot(t_radar[peaks_sound], heart_sounds[peaks_sound], 'X')

# heart_sounds_filt_x = heart_sounds * np.sin(t_radar*2*np.pi*60)
# heart_sounds_filt_y = heart_sounds * np.cos(t_radar*2*np.pi*60)
# sos_heart_sounds_filt = signal.butter(4, 5, 'lowpass', fs=fs_radar, output='sos')
# heart_sounds_filt_x = signal.sosfiltfilt(sos_heart_sounds_filt, heart_sounds_filt_x)
# heart_sounds_filt_y = signal.sosfiltfilt(sos_heart_sounds_filt, heart_sounds_filt_y)
# heart_sounds_filt = np.sqrt(heart_sounds_filt_x**2 + heart_sounds_filt_y**2)

# plt.figure()
# plt.plot(t_radar, heart_sounds/max(abs(heart_sounds)))
# plt.plot(t_radar[200:], heart_sounds_filt[200:]/max(abs(heart_sounds_filt[200:])))
# border = ndimage.uniform_filter1d(abs(heart_sounds_filt/max(abs(heart_sounds_filt))), size=500, output=np.float64, mode='nearest')
# plt.plot(t_radar, border)
# peaks, _ = signal.find_peaks(heart_sounds_filt[200:]/max(abs(heart_sounds_filt[200:])), height=border[200:], distance=500)
# plt.plot(t_radar[200:][peaks], heart_sounds_filt[200:][peaks]/max(abs(heart_sounds_filt[200:])), 'X')


# r = signal.correlate(heart_sounds[8000:9000], pulse_wave_swt[8000:9000])
# # r = 0.0005/signal.windows.triang(len(r)) * r
# P = np.fft.fft(r)
# Phi = 1 / np.abs(P)
# g = np.fft.ifft(P*Phi)
#
# plt.figure()
# # g[:10] = 0
# # g[1995:2005] = 0
# # g[-10:] = 0
# plt.plot(r/max(abs(r)))
# plt.plot(g/max(abs(g)))

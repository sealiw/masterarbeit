import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from scipy import signal
from helper_functions import *
import pickle
from robustsp.LocationScale.MLocHub import *

mat = scipy.io.loadmat('displacement_data\GDN0001\GDN0001_1_Resting.mat')

fs_ecg      = mat['fs_ecg'][0][0]
fs_radar    = mat['fs_radar'][0][0]
radar_i     = mat['radar_i'][:, 0]
radar_q     = mat['radar_q'][:, 0]
displacement= mat['displacement'][:, 0]
tfm_ecg1    = mat['tfm_ecg1'][:, 0]
tfm_ecg2    = mat['tfm_ecg2'][:, 0]


t_radar = np.arange(1 / fs_radar, (len(radar_i) + 1) / fs_radar, 1 / fs_radar)
t_ecg = np.arange(1 / fs_ecg, (len(tfm_ecg1) + 1) / fs_ecg, 1 / fs_ecg)

# delta_phi = np.arctan(radar_q / radar_i)
delta_phi = displacement
sos = signal.butter(10, [16, 80], 'bandpass', fs=fs_radar, output='sos')
delta_phi_filt = signal.sosfilt(sos, delta_phi)

# # R_peaks, weights, weighted_signal = find_ecg_peaks(fs_ecg, tfm_ecg2[int(3e4):int(1e6)])
# R_peaks, weights, weighted_signal = find_ecg_peaks(fs_ecg, tfm_ecg2)
# with open('R_peaks_ecg.pkl', 'wb') as outp:
#     pickle.dump(R_peaks, outp, pickle.HIGHEST_PROTOCOL)

with open('R_peaks_ecg.pkl', 'rb') as inp:
    R_peaks = pickle.load(inp)

brk = int(0.1 * fs_radar)
beat_length = int(0.3 * fs_radar)
beat_sounds = np.empty(shape=[len(R_peaks), beat_length])


for i, peak in enumerate(R_peaks):
    # beat_sounds[i, :] = delta_phi_filt[int(3e4):int(1e6)][peak + brk:peak + brk + beat_length]
    try:
        beat_sounds[i, :] = delta_phi_filt[peak + brk:peak + brk + beat_length]
    except ValueError:
        beat_sounds = np.delete(beat_sounds, -1, 0)
        break

beat_sound_mean = np.empty(shape=np.shape(beat_sounds)[1])
for i in range(np.shape(beat_sounds)[1]):
    beat_sound_mean[i] = MLocHUB(beat_sounds[:, i])

plt.figure()
for i, peak in enumerate(R_peaks[:np.shape(beat_sounds)[0]]):
    plt.plot(t_radar[peak + brk:peak + brk + beat_length], beat_sounds[i])
plt.show()

plt.figure()
plt.plot(beat_sound_mean)
plt.show()

with open('heart_sound_mean.pkl', 'wb') as outp:  # Overwrites any existing file.
    pickle.dump(beat_sound_mean, outp, pickle.HIGHEST_PROTOCOL)

plt.pause(100)

import matplotlib.pyplot as plt
import scipy.io
import numpy as np




mat = scipy.io.loadmat('displacement_data\GDN0001\GDN0001_1_Resting.mat')

radar_i          = mat['radar_i'][:, 0]
sequence_indices = np.arange(2e4, 1e6, dtype=int)
# sequence_indices = np.arange(1e3, 1e5, dtype=int)

fs_bp               = mat['fs_bp'][0][0]
fs_ecg              = mat['fs_ecg'][0][0]
fs_icg              = mat['fs_icg'][0][0]
fs_intervention     = mat['fs_intervention'][0][0]
fs_radar            = mat['fs_radar'][0][0]
radar_i             = mat['radar_i'][sequence_indices, 0]
radar_q             = mat['radar_q'][sequence_indices, 0]
displacement        = mat['displacement'][sequence_indices, 0]
tfm_ecg1            = mat['tfm_ecg1'][sequence_indices, 0]
tfm_ecg2            = mat['tfm_ecg2'][sequence_indices, 0]
tfm_intervention    = mat['tfm_intervention'][sequence_indices, 0]

t_radar = np.arange(1 / fs_radar, (len(radar_i)+1) / fs_radar, 1 / fs_radar)
t_ecg = np.arange(1 / fs_ecg, (len(tfm_ecg1)+1) / fs_ecg, 1 / fs_ecg)

delta_phi = np.arctan(radar_q / radar_i)

plt.plot(t_radar, displacement/max(displacement))
plt.plot(t_radar, delta_phi/max(delta_phi))
# plt.plot(t_radar, radar_i/max(radar_i))
# plt.plot(t_radar, radar_q/max(radar_q))


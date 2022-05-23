import scipy.io
import matplotlib.pyplot as plt
import numpy as np


mat = scipy.io.loadmat('displacement_data\GDN0001\GDN0001_1_Resting.mat')

fs_bp               = mat['fs_bp'][0][0]
fs_ecg              = mat['fs_ecg'][0][0]
fs_icg              = mat['fs_icg'][0][0]
fs_intervention     = mat['fs_intervention'][0][0]
fs_radar            = mat['fs_radar'][0][0]
radar_i             = mat['radar_i'][:, 0]
radar_q             = mat['radar_q'][:, 0]
tfm_bp              = mat['tfm_bp'][:, 0]
tfm_ecg1            = mat['tfm_ecg1'][:, 0]
tfm_ecg2            = mat['tfm_ecg2'][:, 0]
tfm_icg             = mat['tfm_icg'][:, 0]
tfm_intervention    = mat['tfm_intervention'][:, 0]

indices = np.linspace(int(1e3), int(3e4), 500, dtype=int)
indices_2 = np.linspace(int(1e3), int(3e4), int(5e2), dtype=int)


rad_i_data = radar_i[indices]
rad_q_data = radar_q[indices]
rad_i_data_contr = radar_i[indices_2]
rad_q_data_contr = radar_q[indices_2]
A = np.column_stack((rad_i_data*rad_q_data, rad_q_data**2, rad_i_data, rad_q_data, np.ones(len(rad_q_data))))
b = np.expand_dims(-rad_i_data**2, axis=1)
x = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(A), A)),np.transpose(A)), b)

a,b,c,d,e = x[:, 0]

I = np.linspace(min(rad_i_data), max(rad_i_data), 10000)
Q1 = -(a*I+d)/2*b + np.sqrt(((a*I+d)/2*b)**2 - (I**2+c*I+e))
Q2 = -(a*I+d)/2*b - np.sqrt(((a*I+d)/2*b)**2 - (I**2+c*I+e))
plt.plot(I, Q1)
plt.plot(I, Q2)
plt.plot(rad_i_data_contr, rad_q_data_contr, 'X')


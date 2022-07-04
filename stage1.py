import scipy.io
import os.path
import os
import numpy as np

def data_acquisition(file='displacement_data\GDN0004\GDN0004_3_Apnea.mat', to_end=False):
    mat = scipy.io.loadmat(file)

    fs_radar                = mat['fs_radar'][0][0]
    fs_ecg                  = mat['fs_ecg'][0][0]
    fs_icg                  = mat['fs_icg'][0][0]
    fs_bp                   = mat['fs_bp'][0][0]
    fs_intervention         = mat['fs_intervention'][0][0]

    radar_i                 = mat['radar_i'][:, 0]
    radar_q                 = mat['radar_q'][:, 0]
    displacement            = mat['displacement'][:, 0]
    tfm_ecg1                = mat['tfm_ecg1'][:, 0]
    tfm_ecg2                = mat['tfm_ecg2'][:, 0]
    tfm_icg                 = mat['tfm_icg'][:, 0]
    tfm_bp                  = mat['tfm_bp'][:, 0]
    tfm_intervention        = mat['tfm_intervention'][:, 0]

    radar_nan = np.logical_or(np.isnan(radar_i), np.isnan(radar_q))
    ecg_nan = np.logical_or(np.isnan(tfm_ecg1), np.isnan(tfm_ecg2))

    if np.logical_or(radar_nan, ecg_nan).any():
        nan_ind_inv = np.invert(np.logical_or(radar_nan, ecg_nan))
        radar_q = radar_q[nan_ind_inv]
        radar_i = radar_i[nan_ind_inv]
        displacement = displacement[nan_ind_inv]
        tfm_ecg1 = tfm_ecg1[nan_ind_inv]
        tfm_ecg2 = tfm_ecg2[nan_ind_inv]

    radar_i_end_time        = (len(radar_i)-1)/fs_radar
    radar_q_end_time        = (len(radar_q)-1)/fs_radar
    displacement_end_time   = (len(displacement)-1)/fs_radar
    ecg1_end_time           = (len(tfm_ecg1)-1)/fs_ecg
    ecg2_end_time           = (len(tfm_ecg2)-1)/fs_ecg
    icg_end_time            = (len(tfm_icg)-1)/fs_icg
    bp_end_time             = (len(tfm_bp)-1)/fs_bp
    interventions_end_time  = (len(tfm_intervention)-1)/fs_intervention

    if radar_i_end_time == radar_q_end_time == displacement_end_time == ecg1_end_time == ecg2_end_time \
            == icg_end_time == bp_end_time == interventions_end_time:
        end_time = radar_i_end_time
    else:
        end_time = min(radar_i_end_time, radar_q_end_time, displacement_end_time, ecg1_end_time, ecg2_end_time,
                       icg_end_time, bp_end_time, interventions_end_time)

    if to_end:
        radar_start_ind = 0
        radar_indices           = np.arange(radar_start_ind, end_time*fs_radar, dtype=int)
        ecg_indices             = np.arange(radar_start_ind*fs_ecg/fs_radar, end_time*fs_ecg, dtype=int)
        bp_indices              = np.arange(radar_start_ind*fs_bp/fs_radar, end_time*fs_bp, dtype=int)
        icg_indices             = np.arange(radar_start_ind*fs_icg/fs_radar, end_time*fs_icg, dtype=int)
        intervention_indices    = np.arange(radar_start_ind*fs_intervention, end_time*fs_intervention, dtype=int)
    else:
        radar_start_ind = 0
        if 2e6 <= len(radar_i):
            radar_end_ind = 2e6
        elif 1e6 <= len(radar_i):
            radar_end_ind = 1e6
        elif 9e5 <= len(radar_i):
            radar_end_ind = 9e5
        elif 8e5 <= len(radar_i):
            radar_end_ind = 8e5
        elif 7e5 <= len(radar_i):
            radar_end_ind = 7e5
        elif 6e5 <= len(radar_i):
            radar_end_ind = 6e5
        elif 5e5 <= len(radar_i):
            radar_end_ind = 5e5
        elif 4e5 <= len(radar_i):
            radar_end_ind = 4e5
        elif 3e5 <= len(radar_i):
            radar_end_ind = 3e5
        elif 2e5 <= len(radar_i):
            radar_end_ind = 2e5
        elif 1e5 <= len(radar_i):
            radar_end_ind = 1e5
        else:
            radar_end_ind = 9e4
        # radar_end_ind = 3.97e5
        radar_indices           = np.arange(radar_start_ind, radar_end_ind, dtype=int)
        ecg_indices             = np.arange(radar_start_ind*fs_ecg/fs_radar, radar_end_ind*fs_ecg/fs_radar, dtype=int)
        bp_indices              = np.arange(radar_start_ind*fs_bp/fs_radar, radar_end_ind*fs_bp/fs_radar, dtype=int)
        icg_indices             = np.arange(radar_start_ind*fs_icg/fs_radar, radar_end_ind*fs_icg/fs_radar, dtype=int)
        intervention_indices    = np.arange(radar_start_ind*fs_intervention, radar_end_ind*fs_intervention/fs_radar, dtype=int)

    radar_i                 = radar_i[radar_indices]
    radar_q                 = radar_q[radar_indices]
    displacement            = displacement[radar_indices]
    tfm_bp                  = tfm_bp[bp_indices]
    tfm_ecg1                = tfm_ecg1[ecg_indices]
    tfm_ecg2                = tfm_ecg2[ecg_indices]
    tfm_icg                 = tfm_icg[icg_indices]
    tfm_intervention        = tfm_intervention[intervention_indices]


    t_radar                 = np.arange(1 / fs_radar, (len(radar_i)+1) / fs_radar, 1 / fs_radar)
    t_ecg                   = np.arange(1 / fs_ecg, (len(tfm_ecg1)+1) / fs_ecg, 1 / fs_ecg)
    t_bp                    = np.arange(1 / fs_bp, (len(tfm_bp)+1) / fs_bp, 1 / fs_bp)

    return displacement, tfm_ecg2, tfm_bp,  t_radar, fs_radar, t_ecg, fs_ecg, t_bp, fs_bp

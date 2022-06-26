from stage1 import *
from helper_functions import *
from ecgdetectors import Detectors


displacement, tfm_ecg2, t_radar, fs_radar, t_ecg, fs_ecg = data_acquisition()
sos = signal.butter(4, [1.5, 40], 'bandpass', fs=fs_radar, output='sos')
pcg = signal.sosfiltfilt(sos, displacement)
# plt.plot(t_radar, pcg)

pcg_diff = np.diff(pcg)

interval_length = 1.5
interval_sample_length = int(interval_length * fs_radar)
overlapping = 0.5
overlapping_inds = int(interval_sample_length * overlapping)
step = interval_sample_length - overlapping_inds
last_start_ind = int(np.ceil(len(pcg_diff) / step) * step)
new_signal = np.array([])
new_signal_neg = np.array([])
for i in range(0, last_start_ind, step):
    if i + interval_sample_length < len(pcg_diff):
        sig_interval = signal.windows.triang(interval_sample_length) * pcg_diff[
                                                                       i:i + interval_sample_length]
    else:
        sig_interval = signal.windows.triang(interval_sample_length)[
                       :len(pcg_diff[i:])] * pcg_diff[i:]
    mean = MLocHub.MLocHUB(sig_interval, c=0.5)
    sig_interval = sig_interval - mean
    sig_interval = sig_interval / abs(max(sig_interval))
    sig_interval_neg = sig_interval / abs(min(sig_interval))
    if i != 0:
        if len(sig_interval) < overlapping_inds:
            new_step = len(sig_interval)
            new_signal[-new_step:] = new_signal[-new_step:] + sig_interval[:new_step]
            new_signal_neg[-new_step:] = new_signal_neg[-new_step:] + sig_interval_neg[:new_step]
        else:
            new_signal[-overlapping_inds:] = new_signal[-overlapping_inds:] + sig_interval[:overlapping_inds]
            new_signal_neg[-overlapping_inds:] = new_signal_neg[-overlapping_inds:] + sig_interval_neg[
                                                                                      :overlapping_inds]
            new_signal = np.append(new_signal, sig_interval[overlapping_inds:])
            new_signal_neg = np.append(new_signal_neg, sig_interval_neg[overlapping_inds:])
    else:
        new_signal = np.append(new_signal, sig_interval)
        new_signal_neg = np.append(new_signal_neg, sig_interval_neg)


# detectors = Detectors(fs_radar)
# peaks_radar = np.array(detectors.pan_tompkins_detector(new_signal))
# peaks_radar_neg = np.array(detectors.pan_tompkins_detector(-new_signal))
peaks_radar = np.array(find_ecg_peaks(fs_radar, new_signal, negative=False, visgraph=True))
peaks_radar_neg = np.array(find_ecg_peaks(fs_radar, new_signal_neg, negative=True, visgraph=True))
#
# T_1_p = np.mean(new_signal[peaks_radar])
# T_1_n = np.mean(new_signal_neg[peaks_radar_neg])
#
# peaks_radar_ab_T = peaks_radar[new_signal[peaks_radar] > T_1_p]
# peaks_radar_neg_bel_T = peaks_radar_neg[new_signal_neg[peaks_radar_neg] < T_1_n]
#
# T_2_p = np.mean(new_signal[peaks_radar_ab_T])
# T_2_n = np.mean(new_signal_neg[peaks_radar_neg_bel_T])
#
# P_n = peaks_radar_neg[new_signal_neg[peaks_radar_neg] < T_2_n]

plt.plot(t_radar[:-1], new_signal)
# plt.axhline(y=T_2_p, color='r', linestyle='-')
# plt.axhline(y=T_2_n, color='r', linestyle='-')

plt.plot(t_radar[peaks_radar], new_signal[peaks_radar], 'X')
plt.plot(t_radar[peaks_radar_neg], new_signal[peaks_radar_neg], 'X')


heart_rates = compute_heart_rate(peaks_radar, fs_radar)
heart_rates_neg = compute_heart_rate(peaks_radar_neg, fs_radar)
heart_rates_comb = np.array([])
peaks_final = np.array([], dtype=int)
i, j = 0, 0
while i < (len(peaks_radar)-1) and j < (len(peaks_radar_neg)-1):
    if peaks_radar[i] < peaks_radar_neg[j]:
        heart_rates_comb = np.append(heart_rates_comb, heart_rates[i])
        peaks_final = np.append(peaks_final, peaks_radar[i])
        i = i + 1
    elif peaks_radar[i] > peaks_radar_neg[j]:
        heart_rates_comb = np.append(heart_rates_comb, heart_rates_neg[j])
        peaks_final = np.append(peaks_final, peaks_radar_neg[j])
        j = j + 1
    else:
        heart_rates_comb = np.append(heart_rates_comb, (heart_rates_neg[j] + heart_rates[i]) / 2)
        peaks_final = np.append(peaks_final, peaks_radar[i])
        i = i + 1
        j = j + 1

peaks_final = np.append(peaks_final, peaks_radar[-1])
heart_rates_final = running_robust_location(heart_rates_comb, method='m_loc_hub', n=5, c=0.01)

use_existing_ecg_peaks = False
if use_existing_ecg_peaks:
    with open('R_peaks_ecg.pkl', 'rb') as inp:
        R_peaks_ecg = pickle.load(inp)
else:
    R_peaks_ecg = find_ecg_peaks(fs_ecg, tfm_ecg2, visgraph=True)
    with open('R_peaks_ecg.pkl', 'wb') as outp:
        pickle.dump(R_peaks_ecg, outp, pickle.HIGHEST_PROTOCOL)
heart_rates_ref = compute_heart_rate(R_peaks_ecg, fs_ecg)


# Plot peaks and signal
plt.figure()
plot_peaks(t_radar[:-1], new_signal, peaks_final)
plt.plot(t_radar[R_peaks_ecg], new_signal[R_peaks_ecg], 'X')
plt.title('Detection Signal with Peaks')
plt.xlabel('Time in seconds')
plt.ylabel('Detection Signal')

plt.figure()
plot_heart_rates(t_ecg, heart_rates_ref, R_peaks_ecg, label='IBI from ECG')
plot_heart_rates(t_radar, heart_rates_final, peaks_final, label='IBI from Radar')
plt.legend()
plt.title('Interbeat-Intervals')
plt.xlabel('Time in seconds')
plt.ylabel('Interbeat-Intervals')

####### Evaluation #######

evaluation(t_radar, t_ecg, fs_ecg, peaks_final, R_peaks_ecg, heart_rates_final, heart_rates_ref)


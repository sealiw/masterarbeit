from helper_functions import *
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
from stage1 import *


def compute_heart_rates(displacement, tfm_ecg2, t_radar, fs_radar, t_ecg, fs_ecg):
    use_existing_radar_peaks = False
    use_existing_ecg_peaks = False

    if use_existing_radar_peaks:
        with open('Peaks_radar_mov_av.pkl', 'rb') as inp:
            data = pickle.load(inp)
            detection_signal = data['Detection_signal']
            peaks_radar = data['Peaks']
            peaks_radar_neg = data['Peaks_neg']
    else:
        peaks_radar, peaks_radar_neg, detection_signal = moving_average_method(displacement, fs_radar)

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
    plot_peaks(t_radar, detection_signal, peaks_final)
    plt.plot(t_radar[R_peaks_ecg], detection_signal[R_peaks_ecg], 'X')
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

    return heart_rates_final, heart_rates_ref


if __name__ == '__main__':
    displacement, tfm_ecg2, t_radar, fs_radar, t_ecg, fs_ecg = data_acquisition()
    compute_heart_rates(displacement, tfm_ecg2, t_radar, fs_radar, t_ecg, fs_ecg)
    plt.show()
    print('done')

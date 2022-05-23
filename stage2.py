from helper_functions import *
import matplotlib.pyplot as plt

from stage1 import *
import pywt

def compute_heart_rates():

    method = 'moving_average'
    use_existing_peaks = False

    if method == 'heart_sound_method':

        heart_rates_fil1, heart_rates_fil2, peaks_radar, detection_signal = heart_sound_method(displacement, t_radar, fs_radar, True)

        # Plot matched filter output
        plt.figure()
        # plt.plot(t_ecg[int(3e4):int(3e4)+len(matched_filter_out)], tfm_ecg2[int(3e4):int(3e4)+len(matched_filter_out)]/240)
        # plt.plot(t_ecg[len(matched_filter_out)], tfm_ecg2[len(matched_filter_out)]/240)
        plt.plot(t_radar[peaks_radar], detection_signal[peaks_radar], 'X')
        plt.plot(t_radar, detection_signal[:len(t_radar)])

        # R_peaks_ecg, _, _ = find_ecg_peaks(fs_ecg, tfm_ecg2)
        # with open('R_peaks_ecg.pkl', 'wb') as outp:
        #     pickle.dump(R_peaks_ecg, outp, pickle.HIGHEST_PROTOCOL)

        with open('R_peaks_ecg.pkl', 'rb') as inp:
            R_peaks_ecg = pickle.load(inp)
        heart_rates_ref = compute_heart_rate(R_peaks_ecg, fs_ecg)

        plt.figure()
        # plt.plot(t_radar[R_peaks[1:]], heart_rates)
        # plt.plot(t_radar[np.logical_and(t_radar>t_radar[R_peaks[1]],t_radar<t_radar[R_peaks[-1]])], heart_rates_fil)
        # plt.plot(t_radar[R_peaks[1:]], heart_rates_fil)
        # plt.plot(t_radar[np.logical_and(t_radar>t_radar[R_peaks[1]], t_radar<t_radar[R_peaks[-1]])][1::200], heart_rates_arma_fil, label='ARMA')
        plt.plot(t_radar[np.logical_and(t_radar>t_radar[peaks_radar[1]], t_radar<t_radar[peaks_radar[-1]])][1::200], heart_rates_fil1, label='ARMA + lowpass')
        plt.plot(t_radar[np.logical_and(t_radar>t_radar[peaks_radar[1]], t_radar<t_radar[peaks_radar[-1]])][1::200], heart_rates_fil2, label= 'ARMA + median')
        plt.plot(t_ecg[R_peaks_ecg[1:]], heart_rates_ref, label='Reference')
        plt.legend()

    elif method == 'moving_average':
        if use_existing_peaks:
            with open('Peaks_radar_mov_av.pkl', 'rb') as inp:
                data = pickle.load(inp)
                detection_signal = data['Detection_signal']
                peaks_radar = data['Peaks']
        else:
            peaks_radar, detection_signal = moving_average_method(displacement, fs_radar, robust=False)

        # Plot peaks and signal
        plt.figure()
        plot_peaks(t_radar, detection_signal, peaks_radar)
        plt.plot(t_radar, displacement)

        heart_rates = running_robust_location(compute_heart_rate(peaks_radar, fs_radar), method='m_loc_hub', n=3)

        if use_existing_peaks:
            with open('R_peaks_ecg.pkl', 'rb') as inp:
                R_peaks_ecg = pickle.load(inp)
        else:
            R_peaks_ecg, _, _ = find_ecg_peaks(fs_ecg, tfm_ecg2)
            with open('R_peaks_ecg.pkl', 'wb') as outp:
                pickle.dump(R_peaks_ecg, outp, pickle.HIGHEST_PROTOCOL)
        heart_rates_ref = running_robust_location(compute_heart_rate(R_peaks_ecg, fs_ecg), n=3)

        plt.figure()
        plot_heart_rates(t_ecg, heart_rates_ref, R_peaks_ecg)
        plot_heart_rates(t_radar, heart_rates, peaks_radar)


        ####### Evaluation #######

        evaluation(t_radar, t_ecg, fs_ecg, peaks_radar, R_peaks_ecg, heart_rates, heart_rates_ref)


if __name__ == '__main__':
    displacement, tfm_ecg2, t_radar, fs_radar, t_ecg, fs_ecg = data_acquisition()
    compute_heart_rates()
    plt.show()
    print('done')

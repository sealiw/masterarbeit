from visgraphdetector_main.visgraphdetector import VisGraphDetector
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy import signal, interpolate
import robustsp.DependentData.arma_est_bip_mm as arma_est_bip_mm
import robustsp.LocationScale.MLocHub as MLocHub
import pywt



##### Signal Processing Functions #####

def find_ecg_peaks(fs_ecg, ecg, negative=False):
    ecg_min = min(ecg)
    if ecg_min < 0:
        ecg = ecg - ecg_min
    beta = 0.55
    gamma = 0.5
    lowcut = 4
    M = 2 * fs_ecg
    vgd = VisGraphDetector(fs_ecg)
    if negative:
        R_peaks, weights, weighted_signal = vgd.visgraphdetect(-ecg ** 2, beta=beta, gamma=gamma, lowcut=lowcut, M=M)
    else:
        R_peaks, weights, weighted_signal = vgd.visgraphdetect(ecg ** 2, beta=beta, gamma=gamma, lowcut=lowcut, M=M)
    return R_peaks, weights, weighted_signal


def find_radar_peaks(fs_radar, radar):
    beta = 0.55
    gamma = 0.5
    lowcut = 4
    M = 2 * fs_radar
    vgd = VisGraphDetector(fs_radar)
    R_peaks, weights, weighted_signal = vgd.visgraphdetect(radar, beta=beta, gamma=gamma, lowcut=lowcut, M=M)
    return R_peaks, weights, weighted_signal

def compute_heart_rate(peaks, fs):
    heart_rates = np.empty(len(peaks)-1)
    for i in range(len(peaks)-1):
        # heart_rates[i] = fs/(peaks[i+1]-peaks[i])
        heart_rates[i] = (peaks[i+1]-peaks[i])/fs
    return heart_rates

def running_robust_location(sig, method='median', n=20):
    prev_ind = int(n / 2)
    next_ind = int((n-1) / 2)
    if method == 'median':
        signal_pad = np.pad(sig, (prev_ind, next_ind), 'edge')
        signal_filt = np.array([MLocHub.MLocHUB(signal_pad[i - prev_ind:i + next_ind + 1]) for i in range(prev_ind, len(signal_pad) - next_ind)])
    elif method == 'm_loc_hub':
        signal_pad = np.pad(sig, (prev_ind, next_ind), 'edge')
        signal_filt = np.array([MLocHub.MLocHUB(signal_pad[i - prev_ind:i + next_ind + 1]) for i in range(prev_ind, len(signal_pad) - next_ind)])
    else:
        signal_filt = sig
    return signal_filt


def heart_sound_method(displacement, t_radar, fs_radar, use_existing_peaks=True):

    with open('heart_sound_mean.pkl', 'rb') as inp:
        heart_sound_mean = pickle.load(inp)
    filter_displacement = signal.butter(10, [0.75, 80], 'bandpass', fs=fs_radar, output='sos')
    displacement_fil = signal.sosfilt(filter_displacement, displacement)
    matched_filter_out = np.convolve(displacement_fil, np.flip(heart_sound_mean), 'same')

    if use_existing_peaks:
        with open('Peaks_radar.pkl', 'rb') as inp:
            peaks_radar = pickle.load(inp)
    else:
        peaks_radar, _, _ = find_ecg_peaks(fs_radar, matched_filter_out)
        with open('Peaks_radar.pkl', 'wb') as outp:
            pickle.dump(peaks_radar, outp, pickle.HIGHEST_PROTOCOL)

    heart_rates = compute_heart_rate(peaks_radar, fs_radar)
    f = interpolate.interp1d(t_radar[peaks_radar[1:]], heart_rates)
    heart_rates_interp = f(t_radar[np.logical_and(t_radar > t_radar[peaks_radar[1]], t_radar < t_radar[peaks_radar[-1]])])

    heart_rates_arma_fil = arma_est_bip_mm.arma_est_bip_mm(heart_rates_interp[1::200], 1, 0)['cleaned_signal']

    filter_heart_rates = signal.butter(2, 0.03, 'lowpass', fs=fs_radar / 200, output='sos')
    heart_rates_fil1 = signal.sosfilt(filter_heart_rates, heart_rates_arma_fil)
    heart_rates_fil2 = running_robust_location(heart_rates_arma_fil)
    return heart_rates_fil1, heart_rates_fil2, peaks_radar, matched_filter_out


def moving_average_method(sig, fs_radar, robust=True):
    filter_length = 250
    if robust:
        sig_subtr = sig - running_robust_location(sig, n=filter_length, method='m_loc_hub')
    else:
        x = np.pad(sig, (int(filter_length / 2), int((filter_length - 1) / 2)), 'edge')
        sig_subtr = sig - np.convolve(x, np.ones(filter_length) / filter_length, 'valid')

    haar = pywt.Wavelet('haar')
    wvlt_displacement = pywt.swt(sig, haar, 5)



    filter_displacement = signal.butter(10, 20, 'lowpass', fs=fs_radar, output='sos')
    sig_fil = signal.sosfilt(filter_displacement, sig_subtr)


    peaks_radar, _, _ = find_ecg_peaks(fs_radar, sig_fil, negative=True)
    with open('Peaks_radar_mov_av.pkl', 'wb') as outp:
        pickle.dump({'Peaks': peaks_radar, 'Detection_signal': sig_fil}, outp, pickle.HIGHEST_PROTOCOL)

    return peaks_radar, sig_fil


def evaluation(t_radar, t_ecg, fs_ecg, peaks_radar, R_peaks_ecg, heart_rates, heart_rates_ref):
    if t_radar[peaks_radar[-1]] > t_ecg[R_peaks_ecg[-1]]:
        if t_radar[peaks_radar[1]] > t_ecg[R_peaks_ecg[1]]:
            interp_times = t_ecg[np.logical_and(t_radar > t_radar[peaks_radar[1]], t_ecg < t_ecg[R_peaks_ecg[-1]])]
        else:
            interp_times = t_ecg[np.logical_and(t_ecg > t_ecg[R_peaks_ecg[1]], t_ecg < t_ecg[R_peaks_ecg[-1]])]
    else:
        if t_radar[peaks_radar[1]] > t_ecg[R_peaks_ecg[1]]:
            interp_times = t_radar[np.logical_and(t_radar > t_radar[peaks_radar[1]], t_radar < t_radar[peaks_radar[-1]])]
        else:
            interp_times = t_radar[np.logical_and(t_ecg > t_ecg[R_peaks_ecg[1]], t_radar < t_radar[peaks_radar[-1]])]


    f_radar = interpolate.interp1d(t_radar[peaks_radar[1:]], heart_rates)
    heart_rates_interp = f_radar(interp_times)

    f_ecg = interpolate.interp1d(t_radar[R_peaks_ecg[1:]], heart_rates_ref)
    heart_rates_ref_interp = f_ecg(interp_times)

    absolute_deviation = abs(heart_rates_interp - heart_rates_ref_interp)

    plt.figure()
    plt.plot(interp_times, absolute_deviation)

    plt.figure()
    plt.hist(absolute_deviation[0::fs_ecg], bins=30, range=(0, 0.15))


##### Plotting Functions #####

def plot_radar(time_radar, radar_i, radar_q):
    plt.subplot(2, 1, 1)
    plt.plot(time_radar, radar_i)
    plt.subplot(2, 1, 2)
    plt.plot(time_radar, radar_q)
    plt.show()


def plot_peaks(time, signal, peaks):
    plt.plot(time, signal)
    plt.plot(time[peaks], signal[peaks], 'X')


def plot_heart_rates(time, heart_rates, peaks):
    plt.plot(time[peaks[1:]], heart_rates)




##### Other Functions #####

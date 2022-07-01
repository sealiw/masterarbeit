from visgraphdetector_main.visgraphdetector import VisGraphDetector
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})
import numpy as np
import pickle
from scipy import signal, interpolate, ndimage
import robustsp.LocationScale.MLocHub as MLocHub
import pywt
from ecgdetectors import Detectors


##### Signal Processing Functions #####

def find_ecg_peaks(fs_ecg, ecg, negative=False, visgraph=True):
    if visgraph:
        ecg_min = min(ecg)
        if ecg_min < 0:
            ecg = ecg - ecg_min
        beta = 0.55
        gamma = 0.5
        lowcut = 4
        M = 2 * fs_ecg
        vgd = VisGraphDetector(fs_ecg)
        if negative:
            R_peaks, _, _ = vgd.visgraphdetect(-ecg, beta=beta, gamma=gamma, lowcut=lowcut, M=M)
        else:
            R_peaks, _, _ = vgd.visgraphdetect(ecg, beta=beta, gamma=gamma, lowcut=lowcut, M=M)
    else:
        detectors = Detectors(fs_ecg)
        R_peaks = detectors.pan_tompkins_detector(ecg)
    return R_peaks


def find_peaks(x, t, dist, iters, border_size=None):
    x_env = x
    x_new = t
    if border_size:
        border = ndimage.uniform_filter1d(abs(x_env), size=border_size[0], output=np.float64, mode='nearest') + \
                 border_size[1]
        peaks, _ = signal.find_peaks(x, height=border, distance=dist)
    else:
        peaks, _ = signal.find_peaks(x)
    # plt.figure()
    # plt.plot(t, x)
    # plt.plot(t, border)
    # plt.plot(t[peaks], x[peaks], 'X')
    for i in range(1, iters):
        f = interpolate.interp1d(x_new[peaks], x_env[peaks], kind='linear')
        x_new = x_new[peaks[0]:peaks[-1]]
        x_env = f(x_new)
        peaks, _ = signal.find_peaks(x_env, distance=dist)
    peaks = peaks + np.where(t == x_new[0])[0][0]
    return peaks


def compute_heart_rate(peaks, fs):
    heart_rates = np.empty(len(peaks) - 1)
    for i in range(len(peaks) - 1):
        # heart_rates[i] = fs/(peaks[i+1]-peaks[i])
        heart_rates[i] = (peaks[i + 1] - peaks[i]) / fs
    return heart_rates


def compute_PAT(peaks1, peaks2, fs):
    i = 0
    PATs = np.array([])
    while i < min(len(peaks1), len(peaks2)):
        PAT = peaks1[i] - peaks2[i]
        if PAT < 0:
            peaks1 = np.delete(peaks1, i)
        if PAT > 800:
            peaks2 = np.delete(peaks2, i)
        PATs = np.append(PATs, PAT / fs)
        i = i + 1
    return PATs, peaks1


def running_robust_location(sig, method='m_loc_hub', n=20, c=1.345):
    prev_ind = int(n / 2)
    next_ind = int((n - 1) / 2)
    if method == 'median':
        signal_pad = np.pad(sig, (prev_ind, next_ind), 'edge')
        signal_filt = np.array([np.median(signal_pad[i - prev_ind:i + next_ind + 1]) for i in
                                range(prev_ind, len(signal_pad) - next_ind)])
    elif method == 'm_loc_hub':
        signal_pad = np.pad(sig, (prev_ind, next_ind), 'edge')
        signal_filt = np.array([MLocHub.MLocHUB(signal_pad[i - prev_ind:i + next_ind + 1], c=c) for i in
                                range(prev_ind, len(signal_pad) - next_ind)])
    else:
        signal_filt = sig
    return signal_filt


def moving_average_method(sig, fs_radar):
    # Perform moving-average over signal
    filter_length = 100
    x = np.pad(sig, (int(filter_length / 2), int((filter_length - 1) / 2)), 'edge')
    sig_subtr = sig - np.convolve(x, np.ones(filter_length) / filter_length, 'valid')

    # Filter signal with bandpass-filter
    filter_displacement = signal.butter(10, 25, 'lowpass', fs=fs_radar, output='sos')
    sig_fil = signal.sosfilt(filter_displacement, sig_subtr)

    # Apply Wavelet-transformation
    haar = pywt.Wavelet('haar')
    wvlt_displacement = pywt.swt(sig_fil, haar, level=1, start_level=4)[0][1]

    # Divide signal into 'interval_length' seconds and match maximum and minimum amplitudes with an overlapping.
    # Thus, two signals are created. If peaks have similar amplitudes, peak detection works better.
    overlapping = True
    if overlapping:
        interval_length = 1.5
        interval_sample_length = int(interval_length * fs_radar)
        overlapping = 0.5
        overlapping_inds = int(interval_sample_length * overlapping)
        step = interval_sample_length - overlapping_inds
        last_start_ind = int(np.ceil(len(wvlt_displacement) / step) * step)
        new_signal = np.array([])
        new_signal_neg = np.array([])
        for i in range(0, last_start_ind, step):
            if i + interval_sample_length < len(wvlt_displacement):
                sig_interval = signal.windows.triang(interval_sample_length) * wvlt_displacement[
                                                                               i:i + interval_sample_length]
            else:
                sig_interval = signal.windows.triang(interval_sample_length)[
                               :len(wvlt_displacement[i:])] * wvlt_displacement[i:]
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
    else:
        interval_length = 1
        interval_sample_length = int(interval_length * fs_radar)
        last_start_ind = int(np.ceil(len(wvlt_displacement) / interval_sample_length) * interval_sample_length)
        new_signal = np.array([])
        new_signal_neg = np.array([])
        for i in range(0, last_start_ind, interval_sample_length):
            if i + interval_sample_length < len(wvlt_displacement):
                sig_interval = wvlt_displacement[i:i + interval_sample_length]
            else:
                sig_interval = wvlt_displacement[i:]
            new_signal = np.append(new_signal, sig_interval / max(sig_interval))
            new_signal_neg = np.append(new_signal_neg, sig_interval / (-min(sig_interval)))

    # Find peaks and valleys
    peaks_radar = find_ecg_peaks(fs_radar, new_signal, negative=False, visgraph=True)
    peaks_radar_neg = find_ecg_peaks(fs_radar, new_signal_neg, negative=True, visgraph=True)

    # Save peaks into pickle-file
    with open('Peaks_radar_mov_av.pkl', 'wb') as outp:
        pickle.dump({'Peaks': peaks_radar, 'Peaks_neg': peaks_radar_neg, 'Detection_signal': new_signal}, outp,
                    pickle.HIGHEST_PROTOCOL)

    return peaks_radar, peaks_radar_neg, new_signal


def generalized_cross_correlation(x1, x2):
    g = signal.correlate(x1, x2)
    P = np.fft.fft(g)
    Phi = 1 / abs(P)


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


def plot_heart_rates(time, heart_rates, peaks, label):
    plt.plot(time[peaks[1:]], heart_rates, label=label)


##### Other Functions #####

def evaluation(t_radar, t_ecg, fs_ecg, peaks_radar, R_peaks_ecg, heart_rates, heart_rates_ref):
    if t_radar[peaks_radar[-1]] > t_ecg[R_peaks_ecg[-1]]:
        if t_radar[peaks_radar[1]] > t_ecg[R_peaks_ecg[1]]:
            interp_times = t_ecg[np.logical_and(t_radar > t_radar[peaks_radar[1]], t_ecg < t_ecg[R_peaks_ecg[-1]])]
        else:
            interp_times = t_ecg[np.logical_and(t_ecg > t_ecg[R_peaks_ecg[1]], t_ecg < t_ecg[R_peaks_ecg[-1]])]
    else:
        if t_radar[peaks_radar[1]] > t_ecg[R_peaks_ecg[1]]:
            interp_times = t_radar[
                np.logical_and(t_radar > t_radar[peaks_radar[1]], t_radar < t_radar[peaks_radar[-1]])]
        else:
            interp_times = t_radar[np.logical_and(t_ecg > t_ecg[R_peaks_ecg[1]], t_radar < t_radar[peaks_radar[-1]])]

    f_radar = interpolate.interp1d(t_radar[peaks_radar[1:]], heart_rates)
    heart_rates_interp = f_radar(interp_times)

    f_ecg = interpolate.interp1d(t_radar[R_peaks_ecg[1:]], heart_rates_ref)
    heart_rates_ref_interp = f_ecg(interp_times)

    absolute_deviation = abs(heart_rates_interp - heart_rates_ref_interp)

    plt.figure()
    plt.plot(interp_times, absolute_deviation)
    plt.title('Absolute Deviation of IBI-values')
    plt.xlabel('Time in seconds')
    plt.ylabel('Absolute Deviation of IBI-values')

    plt.figure()
    plt.hist(absolute_deviation[0::fs_ecg], bins=30, range=(0, 0.15))  # absolute_deviation[0::fs_ecg] = 488
    plt.title('Histogram of Absolute Deviations of IBI-values')
    plt.xlabel('Absolute Deviation in seconds')
    plt.ylabel('Count of Absolute Deviations')
    print('Percentage of IBI-values deviating less than 10ms: ',
          np.sum(absolute_deviation[0::fs_ecg] < 0.01) / len(absolute_deviation[0::fs_ecg]))
    print('Percentage of IBI-values deviating less than 50ms: ',
          np.sum(absolute_deviation[0::fs_ecg] < 0.05) / len(absolute_deviation[0::fs_ecg]))

from visgraphdetector_main.visgraphdetector import VisGraphDetector
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})
import numpy as np
from scipy import signal, interpolate, ndimage
import robustsp.LocationScale.MLocHub as MLocHub
import robustsp.Regression.ladlasso as ladlasso
import robustsp.Regression.hublasso as hublasso
from sklearn.linear_model import HuberRegressor
from ecgdetectors import Detectors
import pywt


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
        peaks, _ = signal.find_peaks(x, distance=dist)
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


def find_valleys_and_interm_peaks(x, t, peaks):
    peaks_interm = np.array([], dtype=int)
    lowest_valleys = np.array([], dtype=int)
    for i in range(len(peaks) - 1):
        interval = np.arange(peaks[i], peaks[i + 1], 1)
        # find valley
        lowest_valley = interval[x[interval].argmin()]
        lowest_valleys = np.append(lowest_valleys, lowest_valley)
        # find intermediate peak
        interm_peaks = interval[signal.find_peaks(x[interval])[0]]
        highest_interm_peak = interm_peaks[x[interm_peaks].argmax()]
        peaks_interm = np.append(peaks_interm, highest_interm_peak)

    interval = np.arange(peaks[-1], len(x), 1)
    # find last valley
    lowest_valleys = np.append(lowest_valleys, interval[x[interval].argmin()])
    # find last intermediate peak
    interm_peaks, _ = signal.find_peaks(x[interval])
    if interm_peaks.size == 0:
        highest_interm_peak = interval[x[interval].argmax()]
    else:
        highest_interm_peak = interm_peaks[x[interm_peaks].argmax()]
    peaks_interm = np.append(peaks_interm, highest_interm_peak)
    return lowest_valleys, peaks_interm


def get_valid_valleys_and_interm_peaks(valleys, peaks_interm, peaks_valid):
    valleys_valid = np.array([], dtype=int)
    peaks_interm_valid = np.array([], dtype=int)
    for peak in peaks_valid:
        valleys_valid = np.append(valleys_valid, valleys[abs(valleys - peak).argmin()])
        peaks_interm_valid = np.append(peaks_interm_valid, peaks_interm[abs(peaks_interm - peak).argmin()])
    return valleys_valid, peaks_interm_valid


def compute_IBI(peaks, fs):
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
        else:
            PATs = np.append(PATs, PAT / fs)
            i = i + 1
    PATs[PATs == 0] = min(abs(PATs[PATs > 0]))
    return PATs[:-1], peaks1[:-1]


def compute_PAT_valleys(valid_valleys, peaks_sound, fs_radar):
    PATs = np.array([])
    for valley in valid_valleys:
        PAT = min([diff/fs_radar for diff in peaks_sound-valley if diff > 0])
        PATs = np.append(PATs, PAT)
    return PATs

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


def get_BP_values(bp, t_bp, fs_bp, fs_radar, valid_peaks):
    # SBP_inds = np.array(find_ecg_peaks(fs_bp, bp))
    # DBP_inds = np.array(find_ecg_peaks(fs_bp, bp, negative=True))

    SBP_inds = find_peaks(bp, t_bp, 150, 1)
    DBP_inds = find_peaks(-bp, t_bp, 150, 1)

    SBP_inds_valid = np.array([], dtype=int)
    DBP_inds_valid = np.array([], dtype=int)
    factor = fs_radar / fs_bp
    for peak in valid_peaks:
        SBP_diff = abs(SBP_inds * factor - peak)
        closest_SBP = SBP_inds[SBP_diff.argmin()]
        SBP_inds_valid = np.append(SBP_inds_valid, closest_SBP)

        DBP_diff = abs(DBP_inds * factor - peak)
        closest_DBP = DBP_inds[DBP_diff.argmin()]
        DBP_inds_valid = np.append(DBP_inds_valid, closest_DBP)

    return bp[SBP_inds_valid], bp[DBP_inds_valid]


def get_useful_signals(displacement, fs_radar):
    # Filter Pulse-Wave and Heart-Sound
    sos_pulse_wave = signal.butter(4, [0.75, 15], 'bandpass', fs=fs_radar, output='sos')
    pulse_wave = signal.sosfiltfilt(sos_pulse_wave, displacement)

    sos_heart_sounds = signal.butter(4, [20, 60], 'bandpass', fs=fs_radar, output='sos')
    heart_sounds = signal.sosfiltfilt(sos_heart_sounds, displacement)

    # Perform Wavelet Transform
    haar = pywt.Wavelet('haar')
    pulse_wave_swt = pywt.swt(pulse_wave, haar, level=1)[0][1]

    return pulse_wave_swt, heart_sounds


def linear_regression(SBP, DBP, A1, A2):
    y = np.expand_dims(SBP, axis=1)
    # A = np.stack((np.log(PAT), HR, np.ones(len(PAT))), axis=1)
    # beta1 = HuberRegressor().fit(A, y).coef_
    # beta1 = np.array([beta1[1], beta1[2], beta1[0]])

    beta1 = np.linalg.inv(np.transpose(A1) @ A1) @ np.transpose(A1) @ y
    beta2 = np.linalg.inv(np.transpose(A2) @ A2) @ np.transpose(A2) @ y
    return beta1, beta2

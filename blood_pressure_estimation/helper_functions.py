from visgraphdetector_main.visgraphdetector import VisGraphDetector
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
import numpy as np
from scipy import signal, interpolate, ndimage
import robustsp.LocationScale.MLocHub as MLocHub
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
        elif PAT > 800:
            peaks2 = np.delete(peaks2, i)
        else:
            PATs = np.append(PATs, PAT / fs)
            i = i + 1
    return PATs[:-1], peaks1[:-1]


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


def get_BP_values(bp, fs_bp, fs_radar, valid_peaks):
    SBP_inds = np.array(find_ecg_peaks(fs_bp, bp))
    DBP_inds = np.array(find_ecg_peaks(fs_bp, bp, negative=True))

    SBP_inds_valid = np.array([], dtype=int)
    DBP_inds_valid = np.array([], dtype=int)
    factor = fs_radar / fs_bp
    for peak in valid_peaks:
        SBP_diff = abs(SBP_inds * factor - peak)
        closest_SBP = SBP_inds[np.where(min(SBP_diff) == SBP_diff)[0][0]]
        SBP_inds_valid = np.append(SBP_inds_valid, closest_SBP)

        DBP_diff = abs(DBP_inds * factor - peak)
        closest_DBP = DBP_inds[np.where(min(DBP_diff) == DBP_diff)[0][0]]
        DBP_inds_valid = np.append(DBP_inds_valid, closest_DBP)

    return bp[SBP_inds_valid], bp[DBP_inds_valid]

def get_useful_signals(displacement, fs_radar):
    # Filter Pulse-Wave and Heart-Sound
    sos_pulse_wave = signal.butter(4, [0.75, 15], 'bandpass', fs=fs_radar, output='sos')
    pulse_wave = signal.sosfiltfilt(sos_pulse_wave, displacement)

    sos_heart_sounds = signal.butter(4, [20, 40], 'bandpass', fs=fs_radar, output='sos')
    heart_sounds = signal.sosfiltfilt(sos_heart_sounds, displacement)

    # Perform Wavelet Transform
    haar = pywt.Wavelet('haar')
    pulse_wave_swt = pywt.swt(pulse_wave, haar, level=1)[0][1]

    return pulse_wave_swt, heart_sounds

def linear_regression(SBP, DBP, PAT, HR):
    y = np.expand_dims(SBP, axis=1)
    A = np.stack((np.log(PAT), HR, np.ones(len(PAT))), axis=1)
    beta = np.linalg.inv(np.transpose(A)@A)@np.transpose(A)@y
    return beta

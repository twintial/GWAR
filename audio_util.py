from scipy.signal import butter, filtfilt
import numpy as np


def butter_bandpass_filter(data, lowcut, highcut, fs=48e3, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    return filtfilt(b, a, data)


# butter lowpass
def butter_lowpass_filter(data, cutoff, fs=48e3, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def get_cos_IQ_raw_offset(data: np.ndarray, f, offset, fs=48e3) -> (np.ndarray, np.ndarray):
    frames = data.shape[1]
    # offset会一直增长，可能存在问题
    times = np.arange(offset, offset + frames) * 1 / fs
    I_raw = np.cos(2 * np.pi * f * times) * data
    Q_raw = -np.sin(2 * np.pi * f * times) * data
    return I_raw, Q_raw


def get_phase(I: np.ndarray, Q: np.ndarray) -> np.ndarray:
    signal = I + 1j * Q
    angle = np.angle(signal)
    # 这里的axis要看一下对不对
    unwrap_angle = np.unwrap(angle)
    return unwrap_angle
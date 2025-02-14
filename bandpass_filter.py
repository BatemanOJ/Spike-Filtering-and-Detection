from scipy.signal import butter, filtfilt # Bandpass filter

# Bandpass filter: imports the data to be filtered and the filter parameters: lowcut, highcut and sampling rate
def bandpass_filter(data, lowcut_1, highcut_1, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut_1 / nyquist
    high = highcut_1 / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)

    return filtered_data
import numpy as np


# Creates a window from the d_x_filtered data using an index provided by the detected_spikes_x variable
#  and returns an array of these windows
def create_window(detected_spikes_x, d_x, d_x_filtered):
    windows_d_x = []
    for i, idx in enumerate(detected_spikes_x):
        if idx >= 45 and idx + 45 < len(d_x):
            # Extract window around the spike
            window = d_x_filtered[idx - 30: idx + 50]
            windows_d_x.append(window)
    windows_d_x = np.array(windows_d_x)
    windows_d_x = (((windows_d_x - np.mean(windows_d_x)) / np.std(windows_d_x))+0.5)*1.6

    return windows_d_x
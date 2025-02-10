import numpy as np

# Creates a window from the d_1 signal data using an index provided by the Index variable 
# each window is augmented based on the parameters: negative_h, positive_h, verticle
def create_training_window(Index, Class, length, negative_h, positive_h, verticle, signal, windows_output, labels_ouput):
    
    for i, idx in enumerate(Index):
        if idx >= 45 and idx + 45 < len(length):
            window = signal[idx - negative_h: idx + positive_h]
            windows_output.append(window + verticle)
            labels_ouput.append(Class[i])

    return windows_output, labels_ouput
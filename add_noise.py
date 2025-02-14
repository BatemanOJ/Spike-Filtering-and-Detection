import numpy as np

# Imports a signal and then adds a certain amount of noise to the signal depending on the noise_level imputted to the function
def add_noise(signal, noise_level):
    noise = np.random.normal(0, noise_level * np.std(signal), size=signal.shape)
    noisy_signal = signal + noise
    return noisy_signal
from scipy.signal import find_peaks

# Detects spikes in the imported signal based on the imported parameters: height, prominence, width and distance
def detect_spikes(signal, height, prominence, width, distance):
    peaks, properties = find_peaks(
        signal, 
        height=height,          
        prominence=prominence,  
        width=width,
        distance=distance
    )
    return peaks, properties
import wfdb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# Path to the data directory
data_path = r'C:\Users\alberuashvili\Desktop\new_data'

# Define the time window (in seconds)
time_window = 2
'''
# Function to extract features from ECG signal
def extract_features(signal, time):
    features = {}
    # Feature extraction for RR intervals
    rr_intervals = np.diff(time[np.where(signal > 0)])  # Assuming signal is positive for R-peaks
    if len(rr_intervals) == 0:
        return None
    features['Mean RR'] = np.mean(rr_intervals)
    features['SDNN'] = np.std(rr_intervals)
    features['RMSSD'] = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
    
    # Peak detection for QRS complexes
    peaks, _ = find_peaks(signal, distance=100)
    if len(peaks) < 2:
        return None
    qrs_intervals = np.diff(time[peaks])
    features['Mean QRS Duration'] = np.mean(qrs_intervals)
    features['Mean QRS Amplitude'] = np.mean(signal[peaks])
    
    # ST segment analysis (Assuming it's the part between QRS and T wave)
    # You might need to adjust this depending on your signal quality and application
    st_segments = []
    for i in range(len(peaks)-1):
        st_segment = signal[peaks[i]:peaks[i+1]]
        if len(st_segment) > 0:
            st_segments.extend(st_segment)
    st_segments = np.array(st_segments)
    features['Mean ST Segment Elevation'] = np.mean(st_segments[st_segments > 0])
    features['Mean ST Segment Depression'] = np.mean(st_segments[st_segments < 0])
    
    # T wave analysis (Assuming T wave follows the QRS complex)
    t_waves = []
    for i in range(1, len(peaks)-1):
        t_wave = signal[peaks[i]:peaks[i+1]]
        if len(t_wave) > 0:
            t_waves.extend(t_wave)
    t_waves = np.array(t_waves)
    features['Mean T Wave Amplitude'] = np.mean(t_waves)
    
    # QT Interval
    qt_intervals = np.diff(peaks)
    features['Mean QT Interval'] = np.mean(qt_intervals)
    
    return features
'''
# Function to extract features from ECG signal
def extract_features(signal, time):
    features = {}
    # Feature extraction for RR intervals
    rr_intervals = np.diff(time[np.where(signal > 0)])  # Assuming signal is positive for R-peaks
    if len(rr_intervals) == 0:
        return None
    features['Mean RR'] = np.mean(rr_intervals)
    features['SDNN'] = np.std(rr_intervals)
    features['RMSSD'] = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
    
    # Peak detection for QRS complexes
    peaks, _ = find_peaks(signal, distance=100)
    if len(peaks) < 2:
        return None
    qrs_intervals = np.diff(time[peaks])
    features['Mean QRS Duration'] = np.mean(qrs_intervals)
    features['Mean QRS Amplitude'] = np.mean(signal[peaks])
    
    # ST segment analysis (Assuming it's the part between QRS and T wave)
    # You might need to adjust this depending on your signal quality and application
    st_segments = []
    for i in range(len(peaks)-1):
        st_segment = signal[peaks[i]:peaks[i+1]]
        if len(st_segment) > 0:
            st_segments.extend(st_segment)
    st_segments = np.array(st_segments)
    features['Mean ST Segment Elevation'] = np.mean(st_segments[st_segments > 0])
    features['Mean ST Segment Depression'] = np.mean(st_segments[st_segments < 0])
    
    # T wave analysis (Assuming T wave follows the QRS complex)
    t_waves = []
    for i in range(1, len(peaks)-1):
        t_wave = signal[peaks[i]:peaks[i+1]]
        if len(t_wave) > 0:
            t_waves.extend(t_wave)
    t_waves = np.array(t_waves)
    features['Mean T Wave Amplitude'] = np.mean(t_waves)
    
    # QT Interval
    qt_intervals = np.diff(peaks)
    features['Mean QT Interval'] = np.mean(qt_intervals)
    
    # Additional features for RBBB detection
    # Check for morphology of QRS complex
    # Example: Look for presence of RSR' pattern in leads V1 and V2
    rsr_pattern_v1_v2 = (signal[peaks[0]] < signal[peaks[1]]) and (signal[peaks[2]] > signal[peaks[3]])
    features['RSR\' Pattern in V1/V2'] = 1 if rsr_pattern_v1_v2 else 0
    
    return features

# Load the signal
signals, fields = wfdb.rdsamp(data_path + "\\102")

# Calculate the number of samples in the time window
fs = fields['fs']
num_samples_window = int(time_window * fs)

# Extract the subset of the signal within the time window
subset_signals = signals[:num_samples_window, 0]  # Considering only first lead for simplicity
time = np.linspace(0, time_window, num_samples_window)

# Extract features
features = extract_features(subset_signals, time)

# Create DataFrame for features
data = {
    'Mean RR': [features['Mean RR']],
    'SDNN': [features['SDNN']],
    'RMSSD': [features['RMSSD']],
    'Mean QRS Duration': [features['Mean QRS Duration']],
    'Mean QRS Amplitude': [features['Mean QRS Amplitude']],
    'Mean ST Segment Elevation': [features['Mean ST Segment Elevation']],
    'Mean ST Segment Depression': [features['Mean ST Segment Depression']],
    'Mean T Wave Amplitude': [features['Mean T Wave Amplitude']],
    'Mean QT Interval': [features['Mean QT Interval']],
    'RSR\' Pattern in V1/V2': [features['RSR\' Pattern in V1/V2']]
}

new_df = pd.DataFrame(data)

# Show the DataFrame
print(new_df)

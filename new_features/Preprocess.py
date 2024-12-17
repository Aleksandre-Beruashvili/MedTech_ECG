import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
import pywt

def calculate_qrs_duration(ecg_signal, sampling_rate=512):
    qrs_peaks, _ = find_peaks(ecg_signal, height=0)
    if len(qrs_peaks) > 1:
        qrs_duration = (qrs_peaks[-1] - qrs_peaks[0]) / sampling_rate
    else:
        qrs_duration = 0
    return qrs_duration

def calculate_rsr_ratio(ecg_signal):
    r_peak, _ = find_peaks(ecg_signal, height=0)
    if len(r_peak) > 1 and len(r_peak) < 4:
        r_peak_diff = np.diff(r_peak)
        rsr_ratio = r_peak_diff[1] / r_peak_diff[0]
    else:
        rsr_ratio = 0
    return rsr_ratio

def calculate_amplitudes(ecg_signal):
    r_amplitude = np.max(ecg_signal)
    s_amplitude = np.abs(np.min(ecg_signal))
    return r_amplitude, s_amplitude

def calculate_qrs_complex_morphology(ecg_signal):
    r_peak, _ = find_peaks(ecg_signal, height=0)
    if len(r_peak) > 0:
        r_amplitude = ecg_signal[r_peak[0]]
    else:
        r_amplitude = 0
    return r_amplitude

# Path to the CSV file containing ECG data in a single column
file_path = r'C:\Users\alberuashvili\Desktop\apple_health_export\electrocardiograms\ecg_2024-05-19.csv'

# Load CSV file into a pandas DataFrame, starting from the 14th row
df = pd.read_csv(file_path, skiprows=13, header=None, names=['ECG'])

# Example feature extraction
features = []
for ecg_signal in df['ECG']:
    ecg_signal = np.fromstring(str(ecg_signal), dtype=float, sep=',')  # Convert string to numpy array
    qrs_duration = calculate_qrs_duration(ecg_signal)
    rsr_ratio = calculate_rsr_ratio(ecg_signal)
    r_amplitude, s_amplitude = calculate_amplitudes(ecg_signal)
    r_amplitude_in_qrs = calculate_qrs_complex_morphology(ecg_signal)
    features.append([qrs_duration, rsr_ratio, r_amplitude, s_amplitude, r_amplitude_in_qrs])

# Create a DataFrame with the extracted features
feature_names = ['qrs_duration', 'rsr_ratio', 'r_amplitude', 's_amplitude', 'r_amplitude_in_qrs']
feature_df = pd.DataFrame(features, columns=feature_names)

# Combine the feature DataFrame with the original DataFrame
df = pd.concat([df, feature_df], axis=1)

# Print or further process the DataFrame with extracted features
print(df.head())

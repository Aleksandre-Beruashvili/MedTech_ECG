import os
import numpy as np 
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, iirnotch

# Function to preprocess ECG signal for a single patient
record_path = r'C:\Users\alberuashvili\Desktop\mit-bih-arrhythmia-database-1.0.0'

# Extract patient identifier from the file name
patient_id = os.path.basename(record_path).split('.')[0]

# Read the record
record = wfdb.rdrecord(record_path)

# Extract the signal and the sampling rate
ecg_signal = record.p_signal[:, 0]  # Assuming the ECG signal is in the first channel
fs = record.fs  # Sampling rate

# Bandpass filter
lowcut = 0.5
highcut = 40.0
nyquist = 0.5 * fs
low = lowcut / nyquist
high = highcut / nyquist
b, a = butter(1, [low, high], btype='band')
filtered_signal = filtfilt(b, a, ecg_signal)

# Notch filter to remove powerline noise
notch_freq = 50.0
quality_factor = 30.0
b_notch, a_notch = iirnotch(notch_freq / nyquist, quality_factor)
filtered_signal = filtfilt(b_notch, a_notch, filtered_signal)

# Baseline wander removal using a high-pass filter
baseline_cutoff = 0.5
baseline_low = baseline_cutoff / nyquist
b_baseline, a_baseline = butter(1, baseline_low, btype='high')
cleaned_signal = filtfilt(b_baseline, a_baseline, filtered_signal)

# Normalization
normalized_signal = (cleaned_signal - np.mean(cleaned_signal)) / np.std(cleaned_signal)

# Segmentation (example: 5-second windows with 50% overlap)
window_size = 2560  # Adjusted window size
overlap = int(0.5 * window_size)
segments = [normalized_signal[i:i + window_size] for i in range(0, len(normalized_signal) - window_size + 1, overlap)]

# Convert to DataFrame for further analysis
segments_df = pd.DataFrame(segments)
segments_df['patient_id'] = patient_id  # Add patient ID column

# Function to preprocess ECG data for multiple patients
def preprocess_multiple_patients(data_directory):
    all_patient_data = []  # List to store dataframes for each patient
    # List all files in the data directory
    file_list = os.listdir(data_directory)
    
    # Process each file in the directory
    for file in file_list:
        if file.endswith('.dat'):
            record_name = os.path.splitext(file)[0]  # Get the record name without the extension
            record_path = os.path.join(data_directory, record_name)
            segments_df = preprocess_ecg(record_path)
            all_patient_data.append(segments_df)
    # Concatenate dataframes for all patients into a single dataframe
    all_patient_data_df = pd.concat(all_patient_data, ignore_index=True)
    return all_patient_data_df

# Example usage: assuming 'data_directory' is the directory containing the .dat and .hea files
preprocessed_data = preprocess_multiple_patients(data_directory)
print(preprocessed_data.head())  # Print first few rows of the dataframe

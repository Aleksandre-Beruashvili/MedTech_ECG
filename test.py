import pandas as pd
import numpy as np

# Sampling rate
sampling_rate = 512  # Hz

# Path to the CSV file
file_path = r'C:\Users\alberuashvili\Desktop\apple_health_export\electrocardiograms\ecg_2024-05-16_1.csv'

# Load CSV file into a pandas DataFrame, skipping the first 13 rows
df = pd.read_csv(file_path, skiprows=13, header=None, names=['Voltage'])

# Calculate total duration of ECG data
total_duration = len(df) / sampling_rate

# Calculate timestamps for each data point
timestamps = np.linspace(0, total_duration, len(df))

# Create DataFrame with time and signal columns
ecg_df = pd.DataFrame({'Time': timestamps, 'Signal': df['Voltage']})

# Calculate the start and end times for the middle 2 seconds
middle_start_time = total_duration / 2 - 1  # Subtract 1 to get the start time of the middle 2 seconds
middle_end_time = total_duration / 2 + 1    # Add 1 to get the end time of the middle 2 seconds

# Extract the middle 2 seconds of data
middle_ecg_df = ecg_df[(ecg_df['Time'] >= middle_start_time) & (ecg_df['Time'] <= middle_end_time)]

# Reset index of the extracted data
middle_ecg_df.reset_index(drop=True, inplace=True)

print(middle_ecg_df.head())

import wfdb
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ECG_module import extract_middle_segment


# Path to the data directory
data_path = r'C:\Users\alberuashvili\Desktop\mit-bih-arrhythmia-database-1.0.0'

# Path to the labels Excel file
labels_file = r'C:\Users\alberuashvili\Desktop\MedTech_ECG\ecg_labels.xlsx'

# List all the files in the directory
file_list = os.listdir(data_path)

# Filter out only .dat files
dat_files = [f for f in file_list if f.endswith('.dat')]

# Load record names from labels file
labels_df = pd.read_excel(labels_file)
labels_df['Record Name'] = labels_df['Record Name'].astype(str)  # Convert to string

# Find record names that are present both in labels and data directory
record_names_from_labels = labels_df['Record Name'].tolist()
matching_record_names = list(set(record_names_from_labels) & set([os.path.splitext(file)[0] for file in dat_files]))

time_range = (40, 100)  # Time range in seconds
fs = 360

# Create an empty list to store the results
result_list = []

# Process each ECG signal and plot
for record_name in matching_record_names:
    # Load the signal
    signals, fields = wfdb.rdsamp(os.path.join(data_path, record_name))
    
    # Extract the subset of the signal within the time window
    subset_signals = signals[:650000, 0]  # Considering only first lead for simplicity
    
    middle_segment = extract_middle_segment(subset_signals, fs, time_range)
    
    # Convert middle_segment to a list
    middle_segment_list = middle_segment.tolist()
    
    # Append record name and middle segment to the result list
    result_list.append({'Record Name': record_name, 'Middle Segment': middle_segment_list})

    # # Print record name and plot the middle segment
    # print(record_name, middle_segment_list)
    # plt.figure(figsize=(10, 4))
    # plt.plot(middle_segment_list)
    # plt.title(f'ECG Signal - {record_name}')
    # plt.xlabel('Samples')
    # plt.ylabel('Amplitude')
    # plt.grid()
    # plt.show()

# Create DataFrame from the result list
result_df = pd.DataFrame(result_list)

# Merge result_df with labels_df on the record name
merged_df = pd.merge(labels_df, result_df, on='Record Name')

# Print the merged DataFrame
print(merged_df)

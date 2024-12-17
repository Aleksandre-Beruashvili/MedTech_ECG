
import neurokit2 as nk
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



record_names = list(set([os.path.splitext(file)[0] for file in dat_files]))


# Process each ECG signal and plot
for record_name in record_names:
    # Load the signal
    signals, fields = wfdb.rdsamp(os.path.join(data_path, record_name))

    # Extract the subset of the signal within the time window
    subset_signals = signals[:650000, 0]  # Considering only first lead for simplicity
    print(subset_signals)
    print(record_name)




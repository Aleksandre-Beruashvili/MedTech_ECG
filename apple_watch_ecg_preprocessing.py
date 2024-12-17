from ECG_module import extract_middle_segment
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



# Example usage
data = pd.read_csv(r'C:\Users\alberuashvili\Desktop\apple_health_export\electrocardiograms\ecg_2024-06-03_2.csv', skiprows=13, header=None)
ecg_signal = data.iloc[:, 0].values  # Assuming the signal is in the first column
fs = 512  # Sampling rate
time_range = (15, 30)  # Time range in seconds

new_signal = extract_middle_segment(ecg_signal, fs, time_range)

new_signal = pd.DataFrame(new_signal)

values_list = [value for sublist in new_signal.values.tolist() for value in sublist]

new_signal = values_list

print(new_signal)
# Plot the middle segment
plt.figure(figsize=(10, 4))
plt.plot(range(len(new_signal)), new_signal, color='blue')
plt.title('Middle Segment of Filtered ECG Signal')
plt.xlabel('Samples')
plt.ylabel('Normalized Amplitude')
plt.grid(True)
plt.show()


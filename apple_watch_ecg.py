from test import middle_ecg_df
import pandas as pd
import numpy as np
from scipy.signal import find_peaks


# Function to extract features from ECG signal DataFrame
def extract_features(df):
    features = {}
    
    # Feature extraction for RR intervals
    rr_intervals = np.diff(df[df['Signal'] > 0]['Time'])  # Assuming signal is positive for R-peaks
    if len(rr_intervals) == 0:
        return None
    features['Mean RR'] = np.mean(rr_intervals)
    features['SDNN'] = np.std(rr_intervals)
    features['RMSSD'] = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
    
    # Peak detection for QRS complexes
    peaks, _ = find_peaks(df['Signal'], distance=100)
    if len(peaks) < 2:
        return None
    qrs_intervals = np.diff(df['Time'].iloc[peaks])
    features['Mean QRS Duration'] = np.mean(qrs_intervals)
    features['Mean QRS Amplitude'] = np.mean(df['Signal'].iloc[peaks])
    
    # ST segment analysis (Assuming it's the part between QRS and T wave)
    # You might need to adjust this depending on your signal quality and application
    st_segments = []
    for i in range(len(peaks)-1):
        st_segment = df['Signal'].iloc[peaks[i]:peaks[i+1]]
        if len(st_segment) > 0:
            st_segments.extend(st_segment)
    st_segments = np.array(st_segments)
    features['Mean ST Segment Elevation'] = np.mean(st_segments[st_segments > 0])
    features['Mean ST Segment Depression'] = np.mean(st_segments[st_segments < 0])
    
    # T wave analysis (Assuming T wave follows the QRS complex)
    t_waves = []
    for i in range(1, len(peaks)-1):
        t_wave = df['Signal'].iloc[peaks[i]:peaks[i+1]]
        if len(t_wave) > 0:
            t_waves.extend(t_wave)
    t_waves = np.array(t_waves)
    features['Mean T Wave Amplitude'] = np.mean(t_waves)
    
    # QT Interval
    qt_intervals = np.diff(peaks)
    features['Mean QT Interval'] = np.mean(qt_intervals)

    # Check for morphology of QRS complex
    # Example: Look for presence of RSR' pattern in leads V1 and V2
    signal = df['Signal'].values
    if len(peaks) >= 4:  # Ensure there are enough peaks for lead V1 and V2
        rsr_pattern_v1_v2 = (signal[peaks[0]] < signal[peaks[1]]) and (signal[peaks[2]] > signal[peaks[3]])
        features['RSR\' Pattern in V1/V2'] = 1 if rsr_pattern_v1_v2 else 0
    else:
        features['RSR\' Pattern in V1/V2'] = None  # Insufficient peaks for analysis
    
    
    return features


print(middle_ecg_df)
# Extract features from the entire middle_ecg_df DataFrame
features = extract_features(middle_ecg_df)

# If features are not None, append them to the data list
if features is not None:
    data = [features]
else:
    data = []

# Convert data to DataFrame
apple_watch_ecg_df = pd.DataFrame(data)

# Show the DataFrame
print(apple_watch_ecg_df)

print(middle_ecg_df)
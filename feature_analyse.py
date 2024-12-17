from Dat_Hea_preprocessing import preprocessed_data as ecg_df

import pandas as pd
import numpy as np
import scipy.signal

# Heart Rate
def calculate_heart_rate(rr_intervals):
    return 60 / np.mean(rr_intervals)

# Rhythm Regularity
def check_rhythm_regularity(rr_intervals, threshold=0.1):
    return np.std(rr_intervals) < threshold

# QRS Complex Duration
def calculate_qrs_duration(qrs_intervals):
    return np.mean(qrs_intervals)

# Presence of P Waves
def detect_p_waves(ecg_signal, p_wave_threshold=0.1):
    return np.any(ecg_signal > p_wave_threshold)

# PR Interval
def calculate_pr_interval(pr_intervals):
    return np.mean(pr_intervals)

# QT Interval
def calculate_qt_interval(qt_intervals):
    return np.mean(qt_intervals)

# T Wave Morphology
def describe_t_wave_morphology(ecg_signal):
    return "Positive" if np.mean(ecg_signal) > 0 else "Negative"

# Heart Rate Variability (HRV)
def calculate_hrv(rr_intervals):
    return np.std(rr_intervals)

# Presence of Artifacts
def detect_artifacts(ecg_signal, snr_threshold=5):
    return np.mean(ecg_signal) / np.std(ecg_signal) < snr_threshold

# Signal Quality
def assess_signal_quality(ecg_signal):
    return np.var(ecg_signal)

# Dominant Frequency
def calculate_dominant_frequency(ecg_signal, fs):
    freqs, power = scipy.signal.periodogram(ecg_signal, fs)
    return freqs[np.argmax(power)]

# QRS/T Ratio
def calculate_qrs_t_ratio(qrs_duration, qt_interval):
    return qrs_duration / qt_interval

# T Wave Amplitude
def calculate_t_wave_amplitude(ecg_signal):
    return np.max(ecg_signal) - np.min(ecg_signal)

# Heart Rate Trends
def analyze_heart_rate_trends(heart_rate):
    trend = "Increasing" if heart_rate[-1] > heart_rate[0] else "Decreasing" if heart_rate[-1] < heart_rate[0] else "Stable"
    return trend

# Baseline Wander
def detect_baseline_wander(ecg_signal, bw_threshold=0.5):
    return np.max(ecg_signal) - np.min(ecg_signal) > bw_threshold

# Function to calculate features for each segment
def calculate_features_for_segments(data):
    features = []
    for index, row in data.iterrows():
        segment_features = {}
        segment_signal = row.iloc[:-1]  # Extract ECG signal from the row, excluding the last column which contains patient_id
        rr_intervals = calculate_rr_intervals(segment_signal)  # Assuming you have a function to calculate RR intervals
        qrs_intervals = calculate_qrs_intervals(segment_signal)  # Assuming you have a function to calculate QRS intervals
        pr_intervals = calculate_pr_intervals(segment_signal)  # Assuming you have a function to calculate PR intervals
        qt_intervals = calculate_qt_intervals(segment_signal)  # Assuming you have a function to calculate QT intervals

        # Calculate features using the functions defined earlier
        segment_features['heart_rate'] = calculate_heart_rate(rr_intervals)
        segment_features['rhythm_regularity'] = check_rhythm_regularity(rr_intervals)
        segment_features['qrs_duration'] = calculate_qrs_duration(qrs_intervals)
        segment_features['p_waves_present'] = detect_p_waves(segment_signal)
        segment_features['pr_interval'] = calculate_pr_interval(pr_intervals)
        segment_features['qt_interval'] = calculate_qt_interval(qt_intervals)
        segment_features['t_wave_morphology'] = describe_t_wave_morphology(segment_signal)
        segment_features['hrv'] = calculate_hrv(rr_intervals)
        segment_features['artifacts_present'] = detect_artifacts(segment_signal)
        segment_features['signal_quality'] = assess_signal_quality(segment_signal)
        segment_features['dominant_frequency'] = calculate_dominant_frequency(segment_signal, fs)  # fs is the sampling frequency
        segment_features['qrs_t_ratio'] = calculate_qrs_t_ratio(segment_features['qrs_duration'], segment_features['qt_interval'])
        segment_features['t_wave_amplitude'] = calculate_t_wave_amplitude(segment_signal)
        # Add more features as needed

        features.append(segment_features)

    return pd.DataFrame(features)

# Calculate features for segments in the DataFrame
features_df = calculate_features_for_segments(ecg_df)

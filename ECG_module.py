from scipy.signal import butter, filtfilt, iirnotch
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import find_peaks


def ECG_Normalisation(fs,ecg_signal):

    # Bandpass filter
    lowcut = 0.5
    highcut = 40
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
    return normalized_signal


def extract_middle_segment(ecg_signal, fs, time_range):
    # Normalize the ECG signal using the custom function
    normalized_signal = ECG_Normalisation(fs, ecg_signal)

    # Convert time range to sample indices
    start_sample = int(time_range[0] * fs)
    end_sample = int(time_range[1] * fs)

    # Filter the normalized signal with the time range
    filtered_signal = normalized_signal[start_sample:end_sample]

    # Detect R-peaks in the filtered signal
    _, results = nk.ecg_peaks(filtered_signal, sampling_rate=fs)
    rpeaks = results["ECG_R_Peaks"]

    # Select the middle segment
    middle_segment_index = len(rpeaks) // 2
    start_index = rpeaks[middle_segment_index - 5]  # Start of the segment
    end_index = rpeaks[middle_segment_index + 5]    # End of the segment

    middle_segment = filtered_signal[start_index:end_index]

    return middle_segment



# Function to extract features from ECG signal
def ECG_Features_Extraction(signal, time):
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
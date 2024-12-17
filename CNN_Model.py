from ecg_data import merged_df
from apple_watch_ecg_preprocessing import new_signal
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout # type: ignore
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Assuming merged_df is already imported and contains the necessary data
labels = merged_df.drop(['Record Name', 'Middle Segment'], axis=1)
ecg_signals = merged_df['Middle Segment'].tolist()

# Define the desired length for interpolation
target_length = 500  # or another length based on your requirement

def interpolate_signal(signal, target_length):
    original_length = len(signal)
    # Create an interpolation function
    interp_func = interp1d(np.arange(original_length), signal, kind='linear')
    # Generate new indices for the desired length
    new_indices = np.linspace(0, original_length - 1, target_length)
    # Interpolate signal to the new length
    return interp_func(new_indices)


# Normalize the signals to the range [0, 1]
def normalize_signal(signal):
    min_val = np.min(signal)
    max_val = np.max(signal)
    if max_val == min_val:
        return signal
    return (signal - min_val) / (max_val - min_val)


# Interpolate all ECG signals to the target length
interpolated_ecg_signals = np.array([interpolate_signal(signal, target_length) for signal in ecg_signals])
normalized_ecg_signals = np.array([normalize_signal(signal) for signal in interpolated_ecg_signals])

# Process new ECG signals
# Interpolate and normalize new signals
new_interpolated_ecg_signals = interpolate_signal(new_signal, target_length)
new_normalized_ecg_signals = normalize_signal(new_interpolated_ecg_signals)



X = normalized_ecg_signals
y = labels

# Add an extra dimension to the input to make it suitable for Conv1D
X = np.expand_dims(X, axis=2)

# Define the input shape based on the length of each ECG sample
input_shape = X.shape[1:]

# Build the 1D CNN model
model = Sequential([
    Conv1D(32, kernel_size=5, activation='relu', input_shape=input_shape),
    MaxPooling1D(pool_size=2),
    Conv1D(64, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y.shape[1], activation='sigmoid')  # Sigmoid activation for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



# Train the model on all data
model.fit(X, y, epochs=20, batch_size=64)

# Optionally, you can save the trained model for future use
# model.save("your_model.h5")

# # Convert middle_segment to a list
# middle_segment_list = list(middle_segment)

# # Convert middle_segment to a numpy array
# middle_segment_array = np.array(middle_segment_list)

# Reshape the array to match the input shape expected by the model
new_normalized_ecg_signals = new_normalized_ecg_signals.reshape((1, target_length, 1))

# Make predictions using the trained model
predictions = model.predict(new_normalized_ecg_signals)

print(predictions)
# Print model summary
model.summary()




# Calculate percentages from predicted probabilities
predicted_percentages = predictions * 100

# Define a threshold
threshold = 0.5

# Apply thresholding to predictions to get binary labels
binary_predictions = (predictions > threshold).astype(int)

# Assuming each column in y represents a different class
# You can decode the binary predictions into their corresponding labels
predicted_labels = []
for i in range(binary_predictions.shape[1]):
    if binary_predictions[0][i] == 1:
        # Append tuple of label and percentage to the predicted_labels list
        predicted_labels.append((labels.columns[i], predicted_percentages[0][i]))

# Print the predicted labels with percentages
for label, percentage in predicted_labels:
    print(f"Predicted label: {label}, Percentage: {percentage:.2f}%")


# Sum probabilities across predictions
sum_probabilities = np.sum(predictions, axis=0)

# Calculate percentages
total_predictions = predictions.shape[0]  # Total number of predictions
percentages = (sum_probabilities / total_predictions) * 100

# Print percentages for each label
for i, label in enumerate(labels.columns):
    print(f"Label: {label}, Percentage: {percentages[i]:.2f}%")

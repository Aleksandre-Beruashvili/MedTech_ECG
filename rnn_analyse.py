from Dat_Hea_preprocessing import preprocessed_data as df
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load preprocessed data from Dat_Hea_preprocessing.py
from Dat_Hea_preprocessing import preprocessed_data as df

# Path to the labels Excel file
labels_file = r'C:\Users\alberuashvili\Desktop\MedTech_ECG\ecg_labels.xlsx'

# Load record names from labels file
labels_df = pd.read_excel(labels_file)

# Ensure that the 'Record Name' column is string type to match the patient_id format in the preprocessed data
labels_df['Record Name'] = labels_df['Record Name'].astype(str)

# Prepare input data and labels
input_data_rnn = df.drop(columns=['patient_id']).values
patient_ids = df['patient_id'].astype(str).values

# Map patient_ids to labels
record_to_labels = dict(zip(labels_df['Record Name'], labels_df.iloc[:, 1:].values))

# Filter out patients without labels
filtered_indices = [i for i, pid in enumerate(patient_ids) if pid in record_to_labels]
filtered_input_data_rnn = input_data_rnn[filtered_indices]
filtered_patient_ids = patient_ids[filtered_indices]
filtered_labels = np.array([record_to_labels[pid] for pid in filtered_patient_ids])

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(filtered_input_data_rnn, filtered_labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f'Training set: {X_train.shape}, {y_train.shape}')
print(f'Validation set: {X_val.shape}, {y_val.shape}')
print(f'Test set: {X_test.shape}, {y_test.shape}')

# Reshape input data to add the channel dimension (necessary for LSTM input)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define RNN model
def build_rnn_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units=64, return_sequences=True),
        LSTM(units=64),
        Dense(units=num_classes, activation='softmax')  # Use softmax activation for multi-class classification
    ])
    return model

# Number of classes (based on the number of columns in the labels DataFrame, excluding 'Record Name')
num_classes = labels_df.shape[1] - 1

# Build the model
input_shape = (X_train.shape[1], X_train.shape[2])
model = build_rnn_model(input_shape, num_classes)

# Compile the model with appropriate loss function for multi-class classification
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Train the model
num_epochs = 10
batch_size = 32

history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_val, y_val))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)

# Plot training and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Make predictions on the test set
predictions = model.predict(X_test)

# Convert predicted probabilities to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Display some predictions and corresponding true labels
for i in range(10):
    print(f'Prediction: {predicted_labels[i]}, True Label: {np.argmax(y_test[i])}')

# Save the trained model
model.save("trained_rnn_model.h5")

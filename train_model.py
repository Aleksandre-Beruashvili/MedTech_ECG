from ecg_data import merged_df
from apple_watch_ecg import apple_watch_ecg_df
from new_ecg_data import new_df

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load and preprocess your data (assuming it's already loaded in merged_df)
features = merged_df.drop(columns=['Record Name', 'Sinus Rhythm', 'Atrial Arrhythmia', 'AV Block', 'Bundle Branch Block', 'Ventricular Arrhythmia', 'Other Conditions/Artifacts'])
labels = merged_df[['Sinus Rhythm', 'Atrial Arrhythmia', 'AV Block', 'Bundle Branch Block', 'Ventricular Arrhythmia', 'Other Conditions/Artifacts']]

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

# Apply SMOTE to each label
X_train_resampled, y_train_resampled = X_train, y_train.copy()

for label in labels.columns:
    # Determine the number of samples in the minority class
    n_minority_samples = y_train[label].sum()
    
    # Set k_neighbors to a value less than or equal to the number of minority samples
    k_neighbors = min(5, n_minority_samples - 1) if n_minority_samples > 1 else 1
    
    # Apply SMOTE
    sm = SMOTE(sampling_strategy='auto', k_neighbors=k_neighbors, random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train[label])
    
    # Append resampled data
    if isinstance(X_train_resampled, np.ndarray):
        X_train_resampled = np.vstack((X_train_resampled, X_res))
    else:
        X_train_resampled = np.vstack((X_train_resampled, X_res.values))
    
    y_train_resampled = pd.concat([y_train_resampled, pd.DataFrame(y_res, columns=[label])])

# Convert to DataFrame
X_train_resampled = pd.DataFrame(X_train_resampled, columns=features.columns)

# Build the model
model = Sequential()
model.add(Dense(64, input_dim=X_train_resampled.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(y_train_resampled.shape[1], activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_resampled, y_train_resampled, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')

# Make predictions
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

# Print classification report with zero_division parameter
print(classification_report(y_test, y_pred, target_names=labels.columns, zero_division=0))

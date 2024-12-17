import wfdb
import os
import pandas as pd

# Path to the data directory
data_path = r'C:\Users\alberuashvili\Desktop\mit-bih-arrhythmia-database-1.0.0'

# List all the files in the directory
file_list = os.listdir(data_path)

# Filter out only .dat files
dat_files = [f for f in file_list if f.endswith('.dat')]

# Create a list to hold metadata
metadata_list = []

# Loop through each ECG signal
for dat_file in dat_files:
    # Extract the record name (e.g., 100, 101)
    record_name = os.path.splitext(dat_file)[0]
    
    # Read header to get metadata
    record = wfdb.rdheader(os.path.join(data_path, record_name))
    
    # Extract age and gender from Patient Age
    patient_age = record.comments[0]
    age = int(patient_age[:2])
    
    # If age is negative, replace it with the average age
    if age < 0:
        age = None  # Placeholder to be replaced with average age later
    
    # Extract gender from Patient Age
    gender = patient_age[3]
    
    # Extract additional metadata
    additional_metadata = '\n'.join(record.comments[2:])
    
    # Append metadata to list if additional metadata is not empty
    if additional_metadata:
        metadata_list.append({'Record Name': record_name, 'Age': age, 'Gender': gender, 'Additional Metadata': additional_metadata})

# Convert the list of dictionaries to a DataFrame
metadata_df = pd.DataFrame(metadata_list)

# Calculate the average age (excluding None values)
average_age = metadata_df['Age'].mean(skipna=True)

# Replace negative ages with the average age
metadata_df['Age'] = metadata_df['Age'].apply(lambda x: x if x is not None and x >= 0 else average_age)

# Convert ages to integers
metadata_df['Age'] = metadata_df['Age'].astype(int)

# Export DataFrame to Excel
excel_file = 'ecg_metadata.xlsx'
metadata_df.to_excel(excel_file, index=False)

print("Metadata saved to", excel_file)

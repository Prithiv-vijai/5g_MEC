import pandas as pd
import numpy as np
import re

# Load your existing dataset
df = pd.read_csv('../data/dataset.csv')

# Convert Bandwidth and Latency values as before
def convert_bandwidth_to_kbps(bandwidth):
    bandwidth = bandwidth.strip().upper()
    if ' KBPS' in bandwidth:
        return float(bandwidth.replace(' KBPS', ''))
    elif ' MBPS' in bandwidth:
        return float(bandwidth.replace(' MBPS', '')) * 1024
    elif ' KB' in bandwidth:
        return float(bandwidth.replace(' KB', ''))
    elif ' MB' in bandwidth:
        return float(bandwidth.replace(' MB', '')) * 1024
    else:
        raise ValueError(f"Unrecognized bandwidth unit in '{bandwidth}'")

def convert_latency_to_ms(latency):
    latency = latency.strip().upper()
    if ' MS' in latency:
        return float(latency.replace(' MS', ''))
    elif ' S' in latency:
        return float(latency.replace(' S', '')) * 1000
    else:
        raise ValueError(f"Unrecognized latency unit in '{latency}'")

# Apply conversion
df['Required_Bandwidth'] = df['Required_Bandwidth'].apply(convert_bandwidth_to_kbps)
df['Allocated_Bandwidth'] = df['Allocated_Bandwidth'].apply(convert_bandwidth_to_kbps)
df['Latency'] = df['Latency'].apply(convert_latency_to_ms)

# Preprocess the Signal_Strength column to remove the unit 'dBm'
df['Signal_Strength'] = df['Signal_Strength'].apply(lambda x: re.sub(r' dBm', '', x)).astype(float)

# Calculate min and max ranges for each application type
ranges_by_app = df.groupby('Application_Type').agg({
    'Latency': ['min', 'max'],
    'Required_Bandwidth': ['min', 'max'],
    'Allocated_Bandwidth': ['min', 'max'],
    'Signal_Strength': ['min', 'max']
}).reset_index()

# Flatten column names
ranges_by_app.columns = ['_'.join(col).strip() for col in ranges_by_app.columns.values]
ranges_by_app.rename(columns={'Application_Type_': 'Application_Type'}, inplace=True)

def generate_within_range(app_type, num_samples, ranges_by_app, df):
    ranges = ranges_by_app[ranges_by_app['Application_Type'] == app_type].iloc[0]

    if len(df[df['Application_Type'] == app_type]) == 1:
        original_row = df[df['Application_Type'] == app_type].iloc[0]
        data = {
            'Latency': np.random.uniform(original_row['Latency'] * 0.9, original_row['Latency'] * 1.1, num_samples),
            'Required_Bandwidth': np.random.uniform(original_row['Required_Bandwidth'] * 0.9, original_row['Required_Bandwidth'] * 1.1, num_samples),
            'Allocated_Bandwidth': np.random.uniform(original_row['Allocated_Bandwidth'] * 0.9, original_row['Allocated_Bandwidth'] * 1.1, num_samples),
            'Signal_Strength': np.random.uniform(original_row['Signal_Strength'] * 0.9, original_row['Signal_Strength'] * 1.1, num_samples)
        }
    else:
        latency_min = ranges['Latency_min']
        latency_max = ranges['Latency_max']
        bandwidth_req_min = ranges['Required_Bandwidth_min']
        bandwidth_req_max = ranges['Required_Bandwidth_max']
        bandwidth_alloc_min = ranges['Allocated_Bandwidth_min']
        bandwidth_alloc_max = ranges['Allocated_Bandwidth_max']
        signal_strength_min = ranges['Signal_Strength_min']
        signal_strength_max = ranges['Signal_Strength_max']

        data = {
            'Latency': np.random.uniform(latency_min, latency_max, num_samples),
            'Required_Bandwidth': np.random.uniform(bandwidth_req_min, bandwidth_req_max, num_samples),
            'Allocated_Bandwidth': np.random.uniform(bandwidth_alloc_min, bandwidth_alloc_max, num_samples),
            'Signal_Strength': np.random.uniform(signal_strength_min, signal_strength_max, num_samples)
        }

    return pd.DataFrame(data)

# Generate new data for each application type
app_types = df['Application_Type'].unique()
num_samples_per_type = (15000 - len(df)) // len(app_types)

new_data_frames = []
for app_type in app_types:
    new_df = generate_within_range(app_type, num_samples_per_type, ranges_by_app, df)
    new_df['Application_Type'] = app_type
    new_data_frames.append(new_df)

# Combine original data with new data
new_df = pd.concat(new_data_frames, ignore_index=True)
augmented_df = pd.concat([df, new_df], ignore_index=True)

# Reset User_ID to be unique across all rows
def generate_unique_ids(num_records):
    return np.arange(1, num_records + 1)

# Generate new User_IDs for all rows
num_records = len(augmented_df)
new_user_ids = generate_unique_ids(num_records)
augmented_df['User_ID'] = new_user_ids
augmented_df=augmented_df.drop('Resource_Allocation', axis=1)

# Save the augmented dataset
output_file = '../data/augmented_dataset.csv'
augmented_df.to_csv(output_file, index=False)
print(f"File saved successfully as {output_file}")

# Preprocessing the augmented dataset
augmented_df = pd.read_csv(output_file)

# Convert categorical data to numeric
augmented_df['Application_Type'] = augmented_df['Application_Type'].astype('category')
application_type_mapping = dict(enumerate(augmented_df['Application_Type'].cat.categories))
augmented_df['Application_Type'] = augmented_df['Application_Type'].cat.codes

# Print the mapping of numerical values to Application_Type categories
print("Application_Type mapping:")
for num, app_type in application_type_mapping.items():
    print(f"{num}: {app_type}")

# Clean User_ID column to retain only numeric part
augmented_df['User_ID'] = augmented_df['User_ID'].astype(int)

# Apply the efficiency calculation
def calculate_efficiency(df):
    # Group by Application_Type and calculate max values within each group
    max_values = df.groupby('Application_Type').agg({
        'Latency': 'max',
        'Allocated_Bandwidth': 'max'
    }).reset_index()

    # Merge max values back to the original dataframe
    df = df.merge(max_values, on='Application_Type', suffixes=('', '_max'))

    # Calculate Efficiency (allowing values to exceed 100)
    df['Efficiency'] = 100 - (df['Latency'] / df['Latency_max']) * 100 + (df['Allocated_Bandwidth'] / df['Allocated_Bandwidth_max']) * 100

    # Drop the temporary max columns
    df.drop(['Latency_max', 'Allocated_Bandwidth_max'], axis=1, inplace=True)

    return df

augmented_df = calculate_efficiency(augmented_df)
augmented_df= augmented_df.drop('Timestamp', axis=1)

# Save the preprocessed and augmented dataset
output_file_preprocessed = '../data/preprocessed_augmented_dataset.csv'
augmented_df.to_csv(output_file_preprocessed, index=False)
print(f"File saved successfully as {output_file_preprocessed}")

import pandas as pd

# Load the dataset
file_path = '../data/dataset.csv'
df = pd.read_csv(file_path)

# Preprocessing steps
# 1. Remove timestamp column
df.drop(columns=['Timestamp'], inplace=True)

# 2. Make the user id column to contain only the numerical digit
df['User_ID'] = df['User_ID'].str.extract('(\d+)').astype(int)

# 3. Convert application type to numeric data
df['Application_Type'] = df['Application_Type'].astype('category')
application_type_mapping = dict(enumerate(df['Application_Type'].cat.categories))
df['Application_Type'] = df['Application_Type'].cat.codes

# Print the mapping of application types to their numeric codes
print("Application Type Mapping:")
for code, app_type in application_type_mapping.items():
    print(f"{app_type}: {code}")

# 4. Remove the unit from signal strength
df['Signal_Strength'] = df['Signal_Strength'].str.replace(' dBm', '').astype(int)

# 5. Remove the unit from latency
df['Latency'] = df['Latency'].str.replace(' ms', '').astype(int)

# 6. Convert both required and allocated bandwidth data to be in kbps without unit
df['Required_Bandwidth'] = df['Required_Bandwidth'].apply(
    lambda x: float(x.replace(' Mbps', '')) * 1000 if 'Mbps' in x else float(x.replace(' Kbps', ''))
)
df['Allocated_Bandwidth'] = df['Allocated_Bandwidth'].apply(
    lambda x: float(x.replace(' Mbps', '')) * 1000 if 'Mbps' in x else float(x.replace(' Kbps', ''))
)

# 7. Remove percentage symbol from resource allocation
df['Resource_Allocation'] = df['Resource_Allocation'].str.replace('%', '').astype(int)

# Save the preprocessed dataset
preprocessed_file_path = '../data/preprocessed_dataset.csv'
df.to_csv(preprocessed_file_path, index=False)

print("Preprocessing complete. Preprocessed dataset saved as 'preprocessed_dataset.csv'.")

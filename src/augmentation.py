import pandas as pd
import numpy as np

# Function to calculate correlation similarity score
def correlation_similarity_score(original, augmented):
    original_corr = original.corr().values.flatten()
    augmented_corr = augmented.corr().values.flatten()
    return np.corrcoef(original_corr, augmented_corr)[0, 1]

# Load the dataset
df = pd.read_csv('../data/preprocessed_dataset.csv')

# Method for augmenting numerical features with random sampling and noise
def random_sampling_with_noise(data, target_size, random_state=None):
    # Set random seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)

    # Separate numerical and categorical columns
    numerical_features = data.select_dtypes(include=[np.number]).columns
    # Keep Application_Type as it is
    numerical_features = numerical_features.difference(['User_ID', 'Application_Type'])
    
    # Calculate the number of rows to sample per application type
    unique_applications = data['Application_Type'].unique()
    rows_per_application = target_size // len(unique_applications)
    
    augmented_data = pd.DataFrame()
    
    # Sample data for each application type
    for app_type in unique_applications:
        app_data = data[data['Application_Type'] == app_type]
        sampled_data = app_data.sample(rows_per_application, replace=True, random_state=random_state).reset_index(drop=True)
        
        # Add noise to numerical features only
        noise = np.random.normal(0, 1, sampled_data[numerical_features].shape)
        noisy_data = sampled_data[numerical_features] + noise
        
        # Concatenate the noisy data for the current application type
        sampled_data[numerical_features] = noisy_data
        augmented_data = pd.concat([augmented_data, sampled_data], ignore_index=True)
    
    # Reset User_ID column with new unique values
    augmented_data['User_ID'] = range(1, len(augmented_data) + 1)
    
    return augmented_data

# Set a random seed for reproducibility
random_seed = 3

# Augment data using random sampling with noise
target_size = 16000
augmented_random = random_sampling_with_noise(df, target_size, random_state=random_seed)

# Append the existing data to the augmented data
combined_data = augmented_random

# Filter out rows based on the specified conditions
filtered_data = combined_data[
    (combined_data['Required_Bandwidth'] >= 0) & 
    (combined_data['Allocated_Bandwidth'] >= 0) 
]

# Calculate and print correlation similarity scores
score_random = correlation_similarity_score(df.drop(columns=['User_ID']), filtered_data.drop(columns=['User_ID']))
print(f"Correlation Similarity Score - Random Sampling with Noise: {score_random}")

# Print the number of rows in the final dataset
print(f"Number of rows in the final dataset: {len(filtered_data)}")

# Save the filtered combined dataset
filtered_data.to_csv('../data/augmented_dataset.csv', index=False)

print("Filtered and augmented dataset saved to the specified path.")

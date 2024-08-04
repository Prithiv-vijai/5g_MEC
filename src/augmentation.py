import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Function to calculate correlation similarity score
def correlation_similarity_score(original, augmented):
    original_corr = original.corr().values.flatten()
    augmented_corr = augmented.corr().values.flatten()
    return np.corrcoef(original_corr, augmented_corr)[0, 1]

# Load the dataset
df = pd.read_csv('../data/preprocessed_dataset.csv')

# Method for augmenting numerical features with random sampling and noise
def random_sampling_with_noise(data, target_size):
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
        sampled_data = app_data.sample(rows_per_application, replace=True).reset_index(drop=True)
        augmented_data = pd.concat([augmented_data, sampled_data], ignore_index=True)
    
    # Add noise only to numerical features except Application_Type
    noise = np.random.normal(0, 0.01, augmented_data[numerical_features].shape)
    augmented_data[numerical_features] += noise
    
    # Reset User_ID column with new unique values
    augmented_data['User_ID'] = range(1, len(augmented_data) + 1)
    
    return augmented_data

# Augment data using random sampling with noise
target_size = 16000
augmented_random = random_sampling_with_noise(df, target_size)

# Calculate and print correlation similarity scores
score_random = correlation_similarity_score(df.drop(columns=['User_ID']), augmented_random.drop(columns=['User_ID']))
print(f"Correlation Similarity Score - Random Sampling with Noise: {score_random}")

# Save the augmented dataset
augmented_random.to_csv('../data/augmented_dataset.csv', index=False)

print("Augmented dataset saved to the specified path.")

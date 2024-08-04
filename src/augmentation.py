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
    
    # Sample data
    augmented_data = data.sample(target_size, replace=True).reset_index(drop=True)
    
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

# --- Additional Code for Comparison ---

# Load datasets
original_df = pd.read_csv('../data/preprocessed_dataset.csv')
augmented_df = pd.read_csv('../data/augmented_dataset.csv')

# Drop User_ID column for correlation calculations
original_df_no_id = original_df.drop(columns=['User_ID'])
augmented_df_no_id = augmented_df.drop(columns=['User_ID'])

# Calculate correlation matrices
original_corr = original_df_no_id.corr()
augmented_corr = augmented_df_no_id.corr()

# Plot correlation matrices with increased horizontal length
plt.figure(figsize=(18, 8))  # Increase width and height for better visibility

# Plot Original Dataset Correlation Matrix
plt.subplot(1, 2, 1)
sns.heatmap(original_corr, annot=True, cmap='coolwarm', center=0, annot_kws={"size": 8})
plt.title('Original Dataset Correlation Matrix')
plt.xticks(rotation=45, ha='right')  # Rotate x labels for better visibility

# Plot Augmented Dataset Correlation Matrix
plt.subplot(1, 2, 2)
sns.heatmap(augmented_corr, annot=True, cmap='coolwarm', center=0, annot_kws={"size": 8})
plt.title('Augmented Dataset Correlation Matrix')
plt.xticks(rotation=45, ha='right')  # Rotate x labels for better visibility

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('../graphs/visualization/augmented_dataset_comparison.png')
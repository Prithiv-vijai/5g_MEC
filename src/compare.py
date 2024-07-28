import pandas as pd
import matplotlib.pyplot as plt

def plot_efficiency_distribution(df1, df2=None):
    # Combine datasets if a second dataset is provided
    if df2 is not None:
        df = pd.concat([df1, df2], ignore_index=True)
    else:
        df = df1

    # Ensure the Efficiency column exists
    if 'Efficiency' not in df.columns:
        raise ValueError("The 'Efficiency' column is missing from the dataset.")

    # Plot frequency distribution of Efficiency
    plt.figure(figsize=(10, 6))
    plt.hist(df['Efficiency'].dropna(), bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Efficiency')
    plt.ylabel('Frequency')
    plt.title('Frequency Distribution of Efficiency')
    plt.grid(True)
    plt.show()

# Example usage:
# Load your datasets
df1 = pd.read_csv('..\data\preprocessed_augmented_dataset.csv')
df2 = pd.read_csv('..\output\updated_dataset_with_efficiency.csv')  # Optional

# Plot distribution
plot_efficiency_distribution(df1, df2)

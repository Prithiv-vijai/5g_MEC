import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
df1 = pd.read_csv(r'../output/updated_dataset_with_efficiency.csv')  # First dataset
df2 = pd.read_csv(r'../data/preprocessed_augmented_dataset.csv')  # Second dataset

# Check if 'Efficiency' column exists in both datasets
if 'Efficiency' in df1.columns and 'Efficiency' in df2.columns:
    plt.figure(figsize=(12, 6))
    
    # Plot histograms for both datasets
    plt.hist(df1['Efficiency'], bins=10, alpha=0.5, label='Optimised allocation', edgecolor='black')
    plt.hist(df2['Efficiency'], bins=10, alpha=0.5, label='Normal allocation', edgecolor='black')
    
    plt.title('Comparison of Efficiency Distributions')
    plt.xlabel('Efficiency')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # Save the comparison plot
    plt.savefig('../output/efficiency_comparison.png')
    plt.show()
else:
    print("The 'Efficiency' column is missing in one or both datasets.")

import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
df1 = pd.read_csv(r'../data/optimized_dataset.csv')  # First dataset
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
    plt.xlim(0, 200)
    
    # Save the comparison plot
    plt.savefig('../graphs/model_output/efficiency_comparison.png')
    
    plt.figure(figsize=(12, 6))
    
    # Plot histograms for both datasets
    plt.hist(df1['Allocated_B'], bins=10, alpha=0.5, label='Optimised allocation', edgecolor='black')
    plt.hist(df2['Allocated_Bandwidth'], bins=10, alpha=0.5, label='Normal allocation', edgecolor='black')
    
    plt.title('Comparison of Bandwidth Distributions')
    plt.xlabel('Allocated Bandwidth')  # Label for x-axis
    plt.ylabel('Frequency')  # Label for y-axis
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # Save the comparison plot
    plt.savefig('../graphs/model_output/bandwidth_comparison.png')
    print('Comparison graph save to "/output/efficiency_comparison.png"')
else:
    print("The 'Efficiency' column is missing in one or both datasets.")
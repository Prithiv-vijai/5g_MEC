import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
df1 = pd.read_csv(r'../output/updated_dataset_with_efficiency.csv')  # First dataset
df2 = pd.read_csv(r'../data/preprocessed_augmented_dataset.csv')  # Second dataset

# Check if 'Efficiency' column exists in both datasets
if 'Efficiency' in df1.columns and 'Efficiency' in df2.columns:
    plt.figure(figsize=(12, 6))
    
    # Scatter plot for Efficiency
    plt.scatter(df1.index, df1['Efficiency'], alpha=0.5, label='Optimised allocation', c='blue', s=1)
    plt.scatter(df2.index, df2['Efficiency'], alpha=0.5, label='Normal allocation', c='red', s=1)
    
    plt.title('Comparison of Efficiency Distributions')
    plt.xlabel('Sample Index')
    plt.ylabel('Efficiency')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # Save the comparison plot
    plt.savefig('../output/efficiency_comparison.png')
    
    plt.figure(figsize=(12, 6))
    
    # Scatter plot for Allocated Bandwidth
    plt.scatter(df1.index, df1['Allocated_B'], alpha=0.5, label='Optimised allocation', c='blue', s=1)
    plt.scatter(df2.index, df2['Allocated_Bandwidth'], alpha=0.5, label='Normal allocation', c='red', s=1)
    
    plt.title('Comparison of Bandwidth Distributions')
    plt.xlabel('Sample Index')
    plt.ylabel('Allocated Bandwidth')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # Save the comparison plot
    plt.savefig('../output/bandwidth_comparison.png')
    print('Comparison graphs saved to "/output/efficiency_comparison.png" and "/output/bandwidth_comparison.png"')
else:
    print("The 'Efficiency' column is missing in one or both datasets.")

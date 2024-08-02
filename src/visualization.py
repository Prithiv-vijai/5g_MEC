import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the dataset (assuming it's named 'preprocessed_augmented_dataset.csv')
df = pd.read_csv('../data/preprocessed_dataset.csv')

# Create a directory to store the graphs if it doesn't exist
graphs_dir = '../graphs/visualization'
os.makedirs(graphs_dir, exist_ok=True)

# Visualize the average allocated bandwidth for each application type
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Application_Type', y='Allocated_Bandwidth', estimator='mean')
plt.title('Average Allocated Bandwidth for Each Application Type')
plt.xlabel('Application Type')
plt.ylabel('Average Allocated Bandwidth')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'average_allocated_bandwidth.png'))
plt.close()

# Analyze the correlation of numeric features only
numeric_df = df.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numeric Features')
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'correlation_matrix.png'))
plt.close()

# Visualize the average latency for each application type
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Application_Type', y='Latency', estimator='mean')
plt.title('Average Latency for Each Application Type')
plt.xlabel('Application Type')
plt.ylabel('Average Latency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'average_latency.png'))
plt.close()

# Visualize the frequency of each application type
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Application_Type')
plt.title('Frequency of Each Application Type')
plt.xlabel('Application Type')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'application_type_frequency.png'))
plt.close()



print(f"All graphs have been saved to the '{graphs_dir}' directory.")

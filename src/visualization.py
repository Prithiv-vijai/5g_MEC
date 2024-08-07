import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the datasets
original_df = pd.read_csv('../data/preprocessed_dataset.csv')
augmented_df = pd.read_csv('../data/augmented_dataset.csv')

# # Drop User_ID column for the analysis
original_df = original_df.drop(columns=['User_ID'])
augmented_df = augmented_df.drop(columns=['User_ID'])

# Create a directory to store the graphs if it doesn't exist
graphs_dir = '../graphs/visualization'
os.makedirs(graphs_dir, exist_ok=True)

# Function to plot side-by-side graphs
def plot_side_by_side(data1, data2, x, y, title, xlabel, ylabel, filename, estimator='mean'):
    plt.figure(figsize=(18, 8))

    # Plot for the original dataset
    plt.subplot(1, 2, 1)
    sns.barplot(data=data1, x=x, y=y, estimator=estimator)
    plt.title('Original Dataset: ' + title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)

    # Plot for the augmented dataset
    plt.subplot(1, 2, 2)
    sns.barplot(data=data2, x=x, y=y, estimator=estimator)
    plt.title('Augmented Dataset: ' + title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, filename))
    plt.close()

# Function to plot the number of rows per application type
def plot_row_count_comparison(data1, data2, filename):
    plt.figure(figsize=(18, 8))

    # Plot for the original dataset
    plt.subplot(1, 2, 1)
    sns.countplot(data=data1, x='Application_Type')
    plt.title('Original Dataset: Number of Rows per Application Type')
    plt.xlabel('Application Type')
    plt.ylabel('Number of Rows')
    plt.xticks(rotation=45)

    # Plot for the augmented dataset
    plt.subplot(1, 2, 2)
    sns.countplot(data=data2, x='Application_Type')
    plt.title('Augmented Dataset: Number of Rows per Application Type')
    plt.xlabel('Application Type')
    plt.ylabel('Number of Rows')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, filename))
    plt.close()

# Visualize the average allocated bandwidth for each application type
plot_side_by_side(original_df, augmented_df, 'Application_Type', 'Allocated_Bandwidth',
                  'Average Allocated Bandwidth for Each Application Type', 'Application Type',
                  'Average Allocated Bandwidth', 'average_allocated_bandwidth.png')

# Analyze the correlation of numeric features only
plt.figure(figsize=(18, 8))

# Plot Original Dataset Correlation Matrix
plt.subplot(1, 2, 1)
sns.heatmap(original_df.corr(), annot=True, cmap='coolwarm')
plt.title('Original Dataset: Correlation Matrix of Numeric Features')
plt.xticks(rotation=45)
plt.yticks(rotation=45)

# Plot Augmented Dataset Correlation Matrix
plt.subplot(1, 2, 2)
sns.heatmap(augmented_df.corr(), annot=True, cmap='coolwarm')
plt.title('Augmented Dataset: Correlation Matrix of Numeric Features')
plt.xticks(rotation=45)
plt.yticks(rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'correlation_matrix.png'))
plt.close()

# Visualize the average latency for each application type
plot_side_by_side(original_df, augmented_df, 'Application_Type', 'Latency',
                  'Average Latency for Each Application Type', 'Application Type',
                  'Average Latency', 'average_latency.png')

# Visualize the frequency of each application type
plt.figure(figsize=(18, 8))

# Plot for the original dataset
plt.subplot(1, 2, 1)
sns.countplot(data=original_df, x='Application_Type')
plt.title('Original Dataset: Frequency of Each Application Type')
plt.xlabel('Application Type')
plt.ylabel('Frequency')
plt.xticks(rotation=45)

# Plot for the augmented dataset
plt.subplot(1, 2, 2)
sns.countplot(data=augmented_df, x='Application_Type')
plt.title('Augmented Dataset: Frequency of Each Application Type')
plt.xlabel('Application Type')
plt.ylabel('Frequency')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'application_type_frequency.png'))
plt.close()

# Visualize the average Signal_Strength for each application type
plot_side_by_side(original_df, augmented_df, 'Application_Type', 'Signal_Strength',
                  'Average Signal Strength for Each Application Type', 'Application Type',
                  'Average Signal Strength', 'average_signal_strength.png')

# Visualize the average Allocated_Bandwidth for each application type
plot_side_by_side(original_df, augmented_df, 'Application_Type', 'Allocated_Bandwidth',
                  'Average Allocated Bandwidth for Each Application Type', 'Application Type',
                  'Average Allocated Bandwidth', 'average_allocated_bandwidth.png')

# Visualize the number of rows per application type
plot_row_count_comparison(original_df, augmented_df, 'application_type_frequency.png')

print(f"All graphs have been saved to the '{graphs_dir}' directory.")

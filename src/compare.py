import pandas as pd
import matplotlib.pyplot as plt
import os

def load_data(old_dataset_path, new_dataset_path):
    """Load the old and new datasets."""
    old_df = pd.read_csv(old_dataset_path)
    new_df = pd.read_csv(new_dataset_path)
    return old_df, new_df

def plot_efficiency_comparison(old_df, new_df, efficiency_column='Efficiency', output_dir='../output'):
    """Plot the efficiency of all users from both datasets and save the plot to the output directory."""
    plt.figure(figsize=(12, 6))
    
    # Plot efficiency for old dataset
    plt.hist(old_df[efficiency_column], bins=30, alpha=0.5, label='Old Dataset', color='blue')
    
    # Plot efficiency for new dataset
    plt.hist(new_df[efficiency_column], bins=30, alpha=0.5, label='New Dataset', color='green')
    
    plt.xlabel('Efficiency')
    plt.ylabel('Number of Users')
    plt.title('Efficiency Distribution in Old and New Datasets')
    plt.legend()
    plt.grid(True)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    plot_file_path = os.path.join(output_dir, 'efficiency_comparison.png')
    plt.savefig(plot_file_path)
    plt.close()
    print(f"[INFO] Efficiency comparison plot saved to {plot_file_path}")

if __name__ == "__main__":
    # Paths to your datasets
    old_dataset_path = '../data/preprocessed_augmented_dataset.csv'  # Old dataset path
    new_dataset_path = '../output/updated_dataset_with_efficiency.csv'  # New dataset path

    # Load datasets
    old_df, new_df = load_data(old_dataset_path, new_dataset_path)
    
    # Plot efficiency comparison and save it to the output folder
    plot_efficiency_comparison(old_df, new_df)

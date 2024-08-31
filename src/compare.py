import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = '../data/model_performance_metrics.csv'
metrics_df = pd.read_csv(file_path)

# Filter HGBRT rows
hgbrt_df = metrics_df[metrics_df['Model Name'].str.contains('Hgbrt|hgbrt')]

# Define metrics to compare, excluding 'Adjusted R2'
metrics = ['MSE', 'RMSE', 'MAE', 'R2', 'MAPE']

# Initialize the plot
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()
fig.suptitle('Comparison of HGBRT Models')

# Plot each metric
for idx, metric in enumerate(metrics):
    if metric in hgbrt_df.columns:
        ax = axes[idx]
        
        # Determine the best value for each metric type
        if metric in ['MSE', 'RMSE', 'MAE']:
            best_value = hgbrt_df[metric].min()
            colors = ['green' if value == best_value else 'red' for value in hgbrt_df[metric]]
        else:  # For 'R2', 'MAPE'
            best_value = hgbrt_df[metric].max()
            colors = ['green' if value == best_value else 'red' for value in hgbrt_df[metric]]
        
        bars = ax.bar(hgbrt_df['Model Name'], hgbrt_df[metric], color=colors)
        ax.set_title(f'{metric} Comparison')
        ax.set_xlabel('Model Name')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        
        # Add values on bars with 4 decimal places
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}', va='bottom', ha='center')  # ha: horizontal alignment

# Remove the subplot for 'Adjusted R2' if it was included
if 'Adjusted R2' not in metrics:
    fig.delaxes(axes[-1])  # Remove the last subplot in the last position

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Print values and ranks with 4 decimal places
print("HGBRT Models Ranking:")
for index, row in hgbrt_df.iterrows():
    print(f"Model: {row['Model Name']}")
    for metric in metrics:
        print(f"  {metric}: {row[metric]:.4f}")

# Save the plot
plt.savefig('../graphs/model_output/hgbrt_all_searches_comparison.png')

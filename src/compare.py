import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = '../data/model_performance_metrics.csv'
metrics_df = pd.read_csv(file_path)

# Filter HGBRT rows
hgbrt_df = metrics_df[metrics_df['Model Name'].str.startswith(('Hgbrt'))]

# Define metrics to compare, excluding 'Adjusted R2'
metrics = ['MSE', 'RMSE', 'MAE', 'R2', 'MAPE']
additional_metric = 'Completion_Time'

# Custom y-limits for each metric
y_limits = {
    'MSE': (0.6, 1.6),    # Adjust the limits based on your dataset
    'RMSE': (0.6, 1.6),
    'MAE': (0.4, 1.2),
    'R2': (0.95, 1),
    'MAPE': (0, 0.015),
    'Completion_Time': (0, hgbrt_df['Completion_Time'].max() * 1.1)  # Adjust based on actual data
}

# Initialize the plot
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()
fig.suptitle('Comparison of HGBRT Models')

# Plot each metric
for idx, metric in enumerate(metrics):
    if metric in hgbrt_df.columns:
        ax = axes[idx]
        
        # Determine the best value for each metric type
        if metric in ['MSE', 'RMSE', 'MAE', 'MAPE']:
            best_value = hgbrt_df[metric].min()
            colors = ['green' if value == best_value else 'red' for value in hgbrt_df[metric]]
        else:  # For 'R2'
            best_value = hgbrt_df[metric].max()
            colors = ['green' if value == best_value else 'red' for value in hgbrt_df[metric]]
        
        bars = ax.bar(hgbrt_df['Model Name'], hgbrt_df[metric], color=colors)
        ax.set_title(f'{metric} Comparison')
        ax.set_xlabel('Model Name')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        
        # Set custom y-limits
        if metric in y_limits:
            ax.set_ylim(y_limits[metric])

        # Add values on bars with 4 decimal places
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}', va='bottom', ha='center')

# Plot 'Completion Time' as an extra subplot
if additional_metric in hgbrt_df.columns:
    ax = axes[-1]  # The last subplot
    best_value = hgbrt_df[additional_metric].min()
    colors = ['green' if value == best_value else 'red' for value in hgbrt_df[additional_metric]]
    
    bars = ax.bar(hgbrt_df['Model Name'], hgbrt_df[additional_metric], color=colors)
    ax.set_title(f'{additional_metric} Comparison')
    ax.set_xlabel('Model Name')
    ax.set_ylabel('Completion Time (s)')  # Assuming time is in seconds
    ax.tick_params(axis='x', rotation=45)
    
    # Set y-limits for Completion Time
    if additional_metric in y_limits:
        ax.set_ylim(y_limits[additional_metric])
    
    # Add values on bars without decimal places
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, f'{int(yval)}', va='bottom', ha='center')

# Adjust layout to avoid overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Print values and ranks with 4 decimal places
print("HGBRT Models Ranking:")
for index, row in hgbrt_df.iterrows():
    print(f"Model: {row['Model Name']}")
    for metric in metrics:
        print(f"  {metric}: {row[metric]:.4f}")
    print(f"  {additional_metric}: {row[additional_metric]:.4f}")

# Save the plot
plt.savefig('../graphs/model_output/hgbrt_all_searches_comparison.png')

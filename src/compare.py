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
    'MSE': (1.2, 1.7),    # Adjust the limits based on your dataset
    'RMSE': (1, 1.4),
    'MAE': (0.8, 1.0),
    'R2': (0.970, 0.985),
    'MAPE': (0.010, 0.014),
    'Completion_Time': (0, hgbrt_df['Completion_Time'].max() * 1.1)  # Adjust based on actual data
}

# Color for all bars (mild blue) and top 3 (green mild)
default_color = 'cornflowerblue'
top3_color = 'mediumseagreen'

# Initialize the plot
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()
fig.suptitle('Comparison of HGBRT Models')

# Plot each metric
for idx, metric in enumerate(metrics):
    if metric in hgbrt_df.columns:
        ax = axes[idx]
        
        # Sort values and get indices of top 3 (best depending on the metric type)
        if metric in ['MSE', 'RMSE', 'MAE', 'MAPE']:
            sorted_df = hgbrt_df.sort_values(by=metric, ascending=True)  # Lower is better
        else:  # For 'R2'
            sorted_df = hgbrt_df.sort_values(by=metric, ascending=False)  # Higher is better
        
        top3_indices = sorted_df.index[:3]
        
        # Assign colors for all bars, top 3 as green
        colors = [
            top3_color if idx in sorted_df.index[:3] else default_color
            for idx in sorted_df.index
        ]
        
        bars = ax.bar(sorted_df['Model Name'], sorted_df[metric], color=colors)
        ax.set_title(f'{metric} Comparison')
        ax.set_xlabel('Model Name')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        
        # Set custom y-limits
        if metric in y_limits:
            ax.set_ylim(y_limits[metric])

        # Add values inside the bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}', va='bottom', ha='center')

# Plot 'Completion Time' as an extra subplot (all blue)
if additional_metric in hgbrt_df.columns:
    ax = axes[-1]  # The last subplot
    sorted_df = hgbrt_df.sort_values(by=additional_metric, ascending=True)  # Lower is better for time
    bars = ax.bar(sorted_df['Model Name'], sorted_df[additional_metric], color=default_color)  # All bars blue
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
        ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', va='bottom', ha='center')

# Adjust layout to avoid overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Print values with 4 decimal places
print("HGBRT Models Performance Metrics:")
for index, row in hgbrt_df.iterrows():
    print(f"Model: {row['Model Name']}")
    for metric in metrics:
        print(f"  {metric}: {row[metric]:.4f}")
    print(f"  {additional_metric}: {row[additional_metric]:.4f}")

# Save the plot
plt.savefig('../graphs/model_output/hgbrt_all_searches_comparison.png')

# This Python script is performing the following tasks:
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = '../data/model_performance_metrics.csv'
metrics_df = pd.read_csv(file_path)

# Filter HGBRT rows
hgbrt_df = metrics_df[metrics_df['Model Name'].str.contains('Light')]

# Define the selected metrics to compare
selected_metrics = ['MSE', 'MAE', 'R2']

# Custom y-limits for each metric
y_limits = {
    'MSE': (0.6, 1.7),    # Adjust the limits based on your dataset
    'MAE': (0.6, 1.0),
    'R2': (0.96, 1),
}

# Color for all bars (mild blue) and top 3 (green mild)
default_color = '#4c72b0'
top3_color = 'mediumseagreen'

# Initialize the plot
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))  # Adjusted for 3 metrics
axes = axes.flatten()
fig.suptitle('Comparison of all Optimization Techniques')

# Plot each selected metric
for idx, metric in enumerate(selected_metrics):
    if metric in hgbrt_df.columns:
        ax = axes[idx]
        
        # Sort values and get indices of top 3 (best depending on the metric type)
        if metric in ['MSE', 'MAE']:
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

# Adjust layout to avoid overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Print values with 4 decimal places
print(" Models Performance Metrics:")
for index, row in hgbrt_df.iterrows():
    print(f"Model: {row['Model Name']}")
    for metric in selected_metrics:
        print(f"  {metric}: {row[metric]:.4f}")

# Save the plot
plt.savefig('../graphs/model_output/all_searches_comparison.png')


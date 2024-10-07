import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data from CSV file
df = pd.read_csv('../data/model_performance_metrics.csv')

# Ensure the column names are correct and match the expected names
df.columns = [col.strip() for col in df.columns]  # Strip any extra spaces

# Function to highlight top model and display values with ranks
def annotate_plot(ax, data, metric):
    bars = ax.patches
    for i, (index, row) in enumerate(data.iterrows()):
        # Format the values
        if metric in ['R2', 'Adjusted R2', 'MAPE']:
            value = f'{row[metric]:.4f}'
        else:
            value = f'{row[metric]:.2f}'
        
        # Calculate annotation positions
        annotation_height = row[metric] + (0.05 if metric != 'MAPE' else 0.01)
        # Annotate with values and rank
        ax.text(i, annotation_height, value, ha='center', va='bottom')
        # Highlight the top model
        if row['Rank'] == 1:
            bars[i].set_facecolor('mediumseagreen')
            bars[i].set_linewidth(2)
        else:
            bars[i].set_facecolor('#4c72b0')

# Get top 5 models for each metric and add ranking
top_models = pd.concat([
    df.nsmallest(5, 'MSE').assign(Metric='MSE'),
    df.nsmallest(5, 'RMSE').assign(Metric='RMSE'),
    df.nsmallest(5, 'MAE').assign(Metric='MAE'),
    df.nlargest(5, 'R2').assign(Metric='R2'),
    df.nsmallest(5, 'MAPE').assign(Metric='MAPE'),
])

# Add ranking column
top_models['Rank'] = top_models.groupby('Metric').cumcount() + 1

# Plotting
fig, axs = plt.subplots(2, 3, figsize=(24, 10))  # Increased figure size

metrics = ['MSE', 'RMSE', 'MAE', 'R2', 'MAPE']
titles = ['Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'Mean Absolute Error (MAE)', 
          'R-squared (R2)', 'Mean Absolute Percentage Error (MAPE)']

# Custom y-axis limits for each metric
y_limits = {
    'MSE': (0, top_models['MSE'].max() + 1),
    'RMSE': (0, 2),
    'MAE': (0, 1.5),
    'R2': (0, 1.25),
    'MAPE': (0, 0.05),  # Increased limit for better visibility
}

for i, metric in enumerate(metrics):
    ax = axs[i // 3, i % 3]
    metric_data = top_models[top_models['Metric'] == metric]
    sns.barplot(data=metric_data, x='Model Name', y=metric, hue='Metric', ax=ax, dodge=False, palette=['lightblue', 'red'])
    ax.set_title(titles[i])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend_.remove()

    # Remove x and y axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Set custom y-axis limit for the metric
    ax.set_ylim(y_limits[metric])

    # Annotate the plot
    annotate_plot(ax, metric_data, metric)

fig.delaxes(axs[1, 2])

# Adjust spacing between subplots
fig.subplots_adjust(hspace=0.4, wspace=0.3)  # Increase space between rows and columns

plt.savefig('../graphs/model_output/top_5_models_with_rank.png')

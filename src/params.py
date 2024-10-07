import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('../data/model_best_params.csv')

# Set BO_TPE as the best model (you can change this logic based on your criteria)
best_model = 'Hgbrt_BO_TPE'

# Define the color for each model, with BO_TPE highlighted
colors = ['mediumseagreen' if model == best_model 
          else 'lightcoral' if model == 'Hgbrt' 
          else '#4c72b0' for model in df['Model Name']]

# Parameters to plot
parameters = ['l2_regularization', 'learning_rate', 'max_depth', 'max_iter', 'max_leaf_nodes', 'min_samples_leaf']

# Set up the plots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

# Plot each parameter in a separate subplot
for i, param in enumerate(parameters):
    ax = axes[i // 3, i % 3]
    
    # Plot the parameter values
    ax.bar(df['Model Name'], df[param], color=colors)
    
    # Highlight the BO_TPE bar
    ax.bar(df.loc[df['Model Name'] == best_model, 'Model Name'], df.loc[df['Model Name'] == best_model, param], color='mediumseagreen')
    
    ax.set_title(param)
    ax.set_xlabel('Model Name')
    ax.set_ylabel(param)
    ax.tick_params(axis='x', rotation=45)

# Adjust the layout
plt.tight_layout()

# Save the plot before showing it
plt.savefig('../graphs/model_output/best_params_comparison.png')
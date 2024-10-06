import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('../data/model_best_params.csv')



# Define the color for each model, with BO_TPE highlighted
colors = 'cornflowerblue'

# Parameters to plot
parameters = ['l2_regularization', 'learning_rate', 'max_depth', 'max_iter', 'max_leaf_nodes', 'min_samples_leaf']

# Set up the plots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

# Plot each parameter in a separate subplot
for i, param in enumerate(parameters):
    ax = axes[i // 3, i % 3]
    
    # Plot the parameter values
    ax.bar(df['Model Name'], df[param], color=colors)
    

    
    ax.set_title(param)
    ax.set_xlabel('Model Name')
    ax.set_ylabel(param)
    ax.tick_params(axis='x', rotation=45)

# Adjust the layout
plt.tight_layout()

# Save the plot before showing it
plt.savefig('../graphs/model_output/best_params_comparison.png')



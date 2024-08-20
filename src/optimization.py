import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# Load the dataset from a CSV file
data = pd.read_csv('../data/augmented_dataset.csv')

# Define features and target
X = data[['Application_Type', 'Signal_Strength', 'Latency', 'Required_Bandwidth', 'Allocated_Bandwidth']]
y = data['Resource_Allocation']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Expanded parameter grid for RandomizedSearchCV
param_grid= {
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
    'max_iter': [100, 200, 300, 400, 500],
    'max_leaf_nodes': [20, 31, 50, 70, 100],
    'max_depth': [None, 5, 10, 15, 20, 25],
    'min_samples_leaf': [10, 20, 30, 40, 50],
    'l2_regularization': [0, 0.01, 0.1, 0.2, 0.5, 1, 2]
}

# Expanded search space for BayesSearchCV
search_space = {
    'learning_rate': Real(0.001, 0.5, prior='uniform'),
    'max_iter': Integer(100, 500),
    'max_leaf_nodes': Integer(20, 100),
    'max_depth': Integer(5, 25) or None,
    'min_samples_leaf': Integer(10, 50),
    'l2_regularization': Real(0, 2, prior='uniform')
}

# Initialize the model
model = HistGradientBoostingRegressor()

# Random Search
random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=50, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

# Bayesian Optimization
bayes_search = BayesSearchCV(model, search_space, n_iter=50, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
bayes_search.fit(X_train, y_train)

# Make predictions
y_pred_random = random_search.best_estimator_.predict(X_test)
y_pred_bayes = bayes_search.best_estimator_.predict(X_test)

# Calculate metrics
def calculate_metrics(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    adjusted_r2 = 1 - (1-r2) * (len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, rmse, r2, adjusted_r2, mape, mae

metrics_random = calculate_metrics(y_test, y_pred_random)
metrics_bayes = calculate_metrics(y_test, y_pred_bayes)

# Store all metrics in a dictionary for comparison
metrics = {
    "Random Search": metrics_random,
    "Bayesian Optimization": metrics_bayes
}

# Extract the metrics for plotting
metric_names = ['MSE', 'RMSE', 'R²', 'Adjusted R²', 'MAPE', 'MAE']
metric_values = np.array([metrics_random, metrics_bayes])
model_names = ['Random Search', 'Bayesian Optimization']

# Plot the metrics
fig, axs = plt.subplots(3, 2, figsize=(15, 12))
axs = axs.ravel()

for i, metric_name in enumerate(metric_names):
    values = metric_values[:, i]
    # Plot the bars
    bars = axs[i].bar(model_names, values, color=['green', 'red'])
    axs[i].set_title(f'{metric_name} Comparison')
    axs[i].set_ylabel(metric_name)
    
    # Display values on top of each bar
    for bar in bars:
        yval = bar.get_height()
        axs[i].text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}', ha='center', va='bottom', fontsize=10, weight='bold')

# Adjust layout for better spacing
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig('../graphs/model_output/hgbrt_with_optimization.png')

# Display the plot
plt.show()

# Print the best-performing models for each metric
for metric_name in metric_names:
    values = metric_values[:, metric_names.index(metric_name)]
    best_model_index = np.argmin(values) if metric_name in ['MSE', 'RMSE', 'MAE'] else np.argmax(values)
    best_model = model_names[best_model_index]
    best_value = values[best_model_index]
    print(f"Best model for {metric_name}: {best_model} with a value of {best_value:.4f}")

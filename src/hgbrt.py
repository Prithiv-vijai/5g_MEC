import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# Load the dataset from a CSV file
data = pd.read_csv('../data/augmented_datasett.csv')

# Define features and target
X = data[['Application_Type', 'Signal_Strength', 'Latency', 'Required_Bandwidth', 'Allocated_Bandwidth']]
y = data['Resource_Allocation']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grids for GridSearchCV and RandomizedSearchCV
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_iter': [100, 200, 300],
    'max_leaf_nodes': [31, 50, 70],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [20, 30, 40],
    'l2_regularization': [0, 0.1, 0.5, 1]
}

# Define the search space for BayesSearchCV
search_space = {
    'learning_rate': Real(0.01, 0.2, prior='uniform'),
    'max_iter': Integer(100, 300),
    'max_leaf_nodes': Integer(31, 70),
    'max_depth': Integer(10, 20) or None,
    'min_samples_leaf': Integer(20, 40),
    'l2_regularization': Real(0, 1, prior='uniform')
}

# Initialize the model
model = HistGradientBoostingRegressor()

# Grid Search
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Random Search
random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=50, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

# Bayesian Optimization
bayes_search = BayesSearchCV(model, search_space, n_iter=50, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
bayes_search.fit(X_train, y_train)

# Fit the default model without any parameter optimization
default_model = HistGradientBoostingRegressor()
default_model.fit(X_train, y_train)

# Make predictions
y_pred_default = default_model.predict(X_test)
y_pred_grid = grid_search.best_estimator_.predict(X_test)
y_pred_random = random_search.best_estimator_.predict(X_test)
y_pred_bayes = bayes_search.best_estimator_.predict(X_test)

# Calculate metrics
def calculate_metrics(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    adjusted_r2 = 1 - (1-r2) * (len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return mse, rmse, r2, adjusted_r2, mape

metrics_default = calculate_metrics(y_test, y_pred_default)
metrics_grid = calculate_metrics(y_test, y_pred_grid)
metrics_random = calculate_metrics(y_test, y_pred_random)
metrics_bayes = calculate_metrics(y_test, y_pred_bayes)

# Store all metrics in a dictionary for comparison
metrics = {
    "Default": metrics_default,
    "Grid Search": metrics_grid,
    "Random Search": metrics_random,
    "Bayesian Optimization": metrics_bayes
}

# Extract the metrics for plotting
metric_names = ['MSE', 'RMSE', 'R²', 'Adjusted R²', 'MAPE']
metric_values = np.array([metrics_default, metrics_grid, metrics_random, metrics_bayes])
model_names = ['Default', 'Grid Search', 'Random Search', 'Bayesian Optimization']

# Plot the metrics
fig, axs = plt.subplots(3, 2, figsize=(15, 12))
axs = axs.ravel()

best_values = []

for i, metric_name in enumerate(metric_names):
    values = metric_values[:, i]
    rank = np.argsort(values) if i < 2 else np.argsort(-values)
    best_value = values[rank[0]]
    best_model = model_names[rank[0]]
    best_values.append((best_model, best_value))
    
    axs[i].bar(model_names, values, color=['blue', 'orange', 'green', 'red'])
    axs[i].set_title(f'{metric_name} Comparison')
    axs[i].set_ylabel(metric_name)
    axs[i].text(rank[0], best_value, f'Best: {best_model}\nValue: {best_value:.4f}', ha='center', va='bottom', color='black', fontsize=10, weight='bold')

# Hide the last empty subplot
axs[-1].axis('off')

# Save the plot as a PNG file
plt.tight_layout()
plt.savefig('../graphs/model_output/hgbrt.png')

# Display the plot
plt.show()

# Print the best values for each metric
for metric_name, (best_model, best_value) in zip(metric_names, best_values):
    print(f"Best {metric_name}: {best_model} with a value of {best_value:.4f}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error

# Load the dataset from a CSV file
data = pd.read_csv('../data/augmented_dataset.csv')

# Define features and target
X = data[['Application_Type', 'Signal_Strength', 'Latency', 'Required_Bandwidth', 'Allocated_Bandwidth']]
y = data['Resource_Allocation']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Expanded parameter grid for GridSearchCV
expanded_param_grid = {
    'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
    'max_iter': [100, 200, 300, 400, 500],
    'max_leaf_nodes': [31, 50, 70, 90, 100],
    'max_depth': [None, 10, 15, 20, 25],
    'min_samples_leaf': [10, 20, 30, 40, 50],
    'l2_regularization': [0, 0.1, 0.5, 1, 2]
}

# Initialize the model
model = HistGradientBoostingRegressor()

# Grid Search with the expanded parameter grid
grid_search = GridSearchCV(model, expanded_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Make predictions with the best estimator found by Grid Search
y_pred_grid = grid_search.best_estimator_.predict(X_test)

# Calculate metrics
def calculate_metrics(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    adjusted_r2 = 1 - (1-r2) * (len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, rmse, r2, adjusted_r2, mape, mae

# Compute metrics for the Grid Search results
metrics_grid = calculate_metrics(y_test, y_pred_grid)

# Store the metrics in a list for easy access
metric_names = ['MSE', 'RMSE', 'R²', 'Adjusted R²', 'MAPE', 'MAE']
metric_values = metrics_grid

# Plot the metrics
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(metric_names, metric_values, color='blue')
ax.set_title('Model Metrics after Grid Search Optimization')
ax.set_ylabel('Metric Value')

# Display values on top of each bar
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}', ha='center', va='bottom', fontsize=10, weight='bold')

# Save the plot as a PNG file
plt.tight_layout()
plt.savefig('../graphs/model_output/hgbrt_grid_search_optimization.png')

# Display the plot
plt.show()

# Print the best-performing model's metrics
for name, value in zip(metric_names, metric_values):
    print(f'{name}: {value:.4f}')

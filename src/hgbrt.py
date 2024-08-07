import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# Load the dataset from a CSV file
data = pd.read_csv('../data/augmented_dataset.csv')

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

# Get the best models
best_grid_model = grid_search.best_estimator_
best_random_model = random_search.best_estimator_
best_bayes_model = bayes_search.best_estimator_

# Make predictions
y_pred_grid = best_grid_model.predict(X_test)
y_pred_random = best_random_model.predict(X_test)
y_pred_bayes = best_bayes_model.predict(X_test)

# Calculate metrics
def calculate_metrics(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    adjusted_r2 = 1 - (1-r2) * (len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return mse, rmse, r2, adjusted_r2, mape

metrics_grid = calculate_metrics(y_test, y_pred_grid)
metrics_random = calculate_metrics(y_test, y_pred_random)
metrics_bayes = calculate_metrics(y_test, y_pred_bayes)

# Print metrics
print("Grid Search Best Parameters:", grid_search.best_params_)
print("Grid Search Metrics - MSE:", metrics_grid[0], "RMSE:", metrics_grid[1], "R2:", metrics_grid[2], "Adjusted R2:", metrics_grid[3], "MAPE:", metrics_grid[4])

print("Random Search Best Parameters:", random_search.best_params_)
print("Random Search Metrics - MSE:", metrics_random[0], "RMSE:", metrics_random[1], "R2:", metrics_random[2], "Adjusted R2:", metrics_random[3], "MAPE:", metrics_random[4])

print("Bayesian Optimization Best Parameters:", bayes_search.best_params_)
print("Bayesian Optimization Metrics - MSE:", metrics_bayes[0], "RMSE:", metrics_bayes[1], "R2:", metrics_bayes[2], "Adjusted R2:", metrics_bayes[3], "MAPE:", metrics_bayes[4])

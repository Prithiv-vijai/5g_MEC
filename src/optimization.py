import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import os 

# Load the dataset from a CSV file
data = pd.read_csv('../data/augmented_dataset.csv')

# Define features and target
X = data[['Application_Type', 'Signal_Strength', 'Latency', 'Required_Bandwidth', 'Allocated_Bandwidth']]
y = data['Resource_Allocation']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to calculate metrics
def calculate_metrics(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    adjusted_r2 = 1 - (1-r2) * (len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, rmse, r2, adjusted_r2, mape, mae

# Function to append metrics to CSV
def append_metrics_to_csv(model_name, metrics, model_category='Boosting Models'):
    # Define the order of columns as per your requirement
    column_order = ['Model Name', 'Model Category', 'MSE', 'RMSE', 'MAE', 'R2', 'MAPE', 'Adjusted R2']
    
    # Create a dictionary with correct keys and values
    metrics_dict = {
        'Model Name': [model_name],
        'Model Category': [model_category],
        'MSE': [metrics[0]],
        'RMSE': [metrics[1]],
        'MAE': [metrics[2]],
        'R2': [metrics[3]],
        'MAPE': [metrics[4]],
        'Adjusted R2': [metrics[5]]
    }
    
    # Convert dictionary to DataFrame
    df_metrics = pd.DataFrame(metrics_dict)
    
    # Append DataFrame to CSV
    file_path = '../data/model_performance_metrics.csv'
    if not os.path.isfile(file_path):
        df_metrics.to_csv(file_path, mode='w', header=True, index=False, columns=column_order)
    else:
        df_metrics.to_csv(file_path, mode='a', header=False, index=False, columns=column_order)
        
# Bayesian Optimization
search_space = {
    'learning_rate': Real(0.001, 0.5, prior='uniform'),
    'max_iter': Integer(100, 500),
    'max_leaf_nodes': Integer(20, 100),
    'max_depth': Integer(5, 25) or None,
    'min_samples_leaf': Integer(10, 50),
    'l2_regularization': Real(0, 2, prior='uniform')
}

model = HistGradientBoostingRegressor()
bayes_search = BayesSearchCV(model, search_space, n_iter=50, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
bayes_search.fit(X_train, y_train)

y_pred_bayes = bayes_search.best_estimator_.predict(X_test)
metrics_bayes = calculate_metrics(y_test, y_pred_bayes)
append_metrics_to_csv('hgbrt_bayesian', metrics_bayes)

# Grid Search
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_iter': [100, 200],
    'max_leaf_nodes': [20, 31],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [20, 30],
    'l2_regularization': [0, 0.1, 0.5]
}

grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

y_pred_grid = grid_search.best_estimator_.predict(X_test)
metrics_grid = calculate_metrics(y_test, y_pred_grid)
append_metrics_to_csv('hgbrt_grid', metrics_grid)

# Random Search
param_grid_random = {
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
    'max_iter': [100, 200, 300, 400, 500],
    'max_leaf_nodes': [20, 31, 50, 70, 100],
    'max_depth': [None, 5, 10, 15, 20, 25],
    'min_samples_leaf': [10, 20, 30, 40, 50],
    'l2_regularization': [0, 0.01, 0.1, 0.2, 0.5, 1, 2]
}

random_search = RandomizedSearchCV(model, param_distributions=param_grid_random, n_iter=50, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

y_pred_random = random_search.best_estimator_.predict(X_test)
metrics_random = calculate_metrics(y_test, y_pred_random)
append_metrics_to_csv('hgbrt_random', metrics_random)




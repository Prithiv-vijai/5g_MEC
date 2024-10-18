import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import os
import time
import optuna
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Load the dataset from a CSV file
data = pd.read_csv('../data/augmented_dataset.csv')

# Define features and target
X = data[['Application_Type', 'Signal_Strength', 'Latency', 'Required_Bandwidth', 'Allocated_Bandwidth']]
y = data['Resource_Allocation']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

iter = 500
state = 42

# Function to calculate metrics
def calculate_metrics(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, rmse, mae, r2, mape

# Append metrics to CSV
def append_metrics_to_csv(model_name, metrics, completion_time, model_category='Tree-Based Models'):
    column_order = ['Model Name', 'Model Category', 'MSE', 'RMSE', 'MAE', 'R2', 'MAPE', 'Completion_Time']
    metrics_dict = {
        'Model Name': [model_name],
        'Model Category': [model_category],
        'MSE': [metrics[0]],
        'RMSE': [metrics[1]],
        'MAE': [metrics[2]],
        'R2': [metrics[3]],
        'MAPE': [metrics[4]],
        'Completion_Time': [completion_time]
    }
    df_metrics = pd.DataFrame(metrics_dict)
    file_path = '../data/model_performance_metrics.csv'
    if not os.path.isfile(file_path):
        df_metrics.to_csv(file_path, mode='w', header=True, index=False, columns=column_order)
    else:
        df_metrics.to_csv(file_path, mode='a', header=False, index=False, columns=column_order)

# Function to evaluate the model with cross-validation
def evaluate_model(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return calculate_metrics(y_test, y_pred)  # Return metrics based on predictions

# Append best parameters to CSV
def append_best_params_to_csv(model_name, best_params):
    for key in best_params:
        if best_params[key] is None:
            best_params[key] = 'None'

    ordered_params = {
        'Model Name': [model_name],
        'n_estimators': [best_params.get('n_estimators', 'None')],
        'max_depth': [best_params.get('max_depth', 'None')],
        'min_samples_leaf': [best_params.get('min_samples_leaf', 'None')],
        'max_leaf_nodes': [best_params.get('max_leaf_nodes', 'None')]
    }

    df_params = pd.DataFrame(ordered_params)
    file_path = '../data/rf_best_params.csv'

    if not os.path.isfile(file_path):
        df_params.to_csv(file_path, mode='w', header=True, index=False)
    else:
        df_params.to_csv(file_path, mode='a', header=False, index=False)

# Global objective function for optimization
def objective_rf(trial):
    n_estimators = trial.suggest_int('n_estimators', 200, 500)
    max_depth = trial.suggest_int('max_depth', 5, 25)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 5, 50)
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 5, 50) 

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        criterion='squared_error',
        max_leaf_nodes=max_leaf_nodes,  
        random_state=42
    )

    # Return the cross-validated mean MSE as the objective value for optimization
    neg_mse_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    mean_mse = -neg_mse_scores.mean()
    return mean_mse


# Bayesian Optimization using TPE
def bayesian_optimization_tpe():
    print("Starting Bayesian Optimization with TPE...")
    start_time = time.time()
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=state))
    study.optimize(objective_rf, n_trials=iter)  # Update to objective_rf
    best_params = study.best_params
    metrics = evaluate_model(RandomForestRegressor(**best_params, random_state=state))

    # Append best metrics and parameters to CSV
    append_metrics_to_csv('RandomForest_BO_TPE', metrics, time.time() - start_time)
    append_best_params_to_csv('RandomForest_BO_TPE', best_params)

# Bayesian Optimization using Gaussian Process
def bayesian_optimization_gp():
    print("Starting Bayesian Optimization with Gaussian Process...")
    start_time = time.time()
    study = optuna.create_study(sampler=optuna.samplers.GPSampler(seed=state))
    study.optimize(objective_rf, n_trials=iter)  # Update to objective_rf
    best_params = study.best_params
    metrics = evaluate_model(RandomForestRegressor(**best_params, random_state=state))

    # Append best metrics and parameters to CSV
    append_metrics_to_csv('RandomForest_BO_GP', metrics, time.time() - start_time)
    append_best_params_to_csv('RandomForest_BO_GP', best_params)
    
def bayesian_optimization_cmaes():
    print("Starting Bayesian Optimization with cmaes...")
    start_time = time.time()
    study = optuna.create_study(sampler=optuna.samplers.CmaEsSampler(seed=state))
    study.optimize(objective_rf, n_trials=iter)  # Update to objective_rf
    best_params = study.best_params
    metrics = evaluate_model(RandomForestRegressor(**best_params, random_state=state))

    # Append best metrics and parameters to CSV
    append_metrics_to_csv('RandomForest_BO_CMAES', metrics, time.time() - start_time)
    append_best_params_to_csv('RandomForest_BO_CMAES', best_params)
    
    
# Function to perform Grid Search
def grid_search_optimization():
    print("Starting Grid Search Optimization...")
    start_time = time.time()
    
    param_grid = {
        'n_estimators': [ 50, 150],
        'max_depth': [10, 15,],
        'min_samples_leaf': [ 1, 10],
        'max_leaf_nodes': [20, 40]
    }
    
    model = RandomForestRegressor(random_state=state)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    y_pred = grid_search.predict(X_train)
    metrics = calculate_metrics(y_train, y_pred)

    # Append best metrics and parameters to CSV
    append_metrics_to_csv('RandomForest_Grid_Search', metrics, time.time() - start_time)
    append_best_params_to_csv('RandomForest_Grid_Search', best_params)


# Function to perform Random Search
def random_search_optimization():
    print("Starting Random Search Optimization...")
    start_time = time.time()
    
    param_distributions = {
        'n_estimators': np.arange(50, 100, 150),
        'max_depth': np.arange(5, 26, 50),
        'min_samples_leaf': np.arange(1, 8, 10),
        'max_leaf_nodes': np.arange(20, 25, 40)
    }
    
    model = RandomForestRegressor(random_state=state)
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, n_iter=50, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=state)
    random_search.fit(X_train, y_train)
    
    best_params = random_search.best_params_
    y_pred = random_search.predict(X_train)
    metrics = calculate_metrics(y_train, y_pred)

    # Append best metrics and parameters to CSV
    append_metrics_to_csv('RandomForest_Random_Search', metrics, time.time() - start_time)
    append_best_params_to_csv('RandomForest_Random_Search', best_params)


# Running the Grid Search and Random Search optimizations
grid_search_optimization()
random_search_optimization()

# Running all optimization techniques
bayesian_optimization_tpe()
bayesian_optimization_gp()
bayesian_optimization_cmaes()


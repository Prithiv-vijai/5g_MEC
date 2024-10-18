import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import os
import time
import optuna
from optuna.samplers import TPESampler, GPSampler, CmaEsSampler
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
def append_metrics_to_csv(model_name, metrics, completion_time, model_category='Boosting Models'):
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
        'num_leaves': [best_params.get('num_leaves', 'None')],
        'n_estimators': [best_params.get('n_estimators', 'None')],
        'learning_rate': [best_params.get('learning_rate', 'None')],
        'max_depth': [best_params.get('max_depth', 'None')],
        'min_data_in_leaf': [best_params.get('min_data_in_leaf', 'None')],
        'lambda_l1': [best_params.get('lambda_l1', 'None')],
        'lambda_l2': [best_params.get('lambda_l2', 'None')],
    }

    df_params = pd.DataFrame(ordered_params)
    file_path = '../data/light_gbm_best_params.csv'

    if not os.path.isfile(file_path):
        df_params.to_csv(file_path, mode='w', header=True, index=False)
    else:
        df_params.to_csv(file_path, mode='a', header=False, index=False)

# Global objective function
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 150) 
    learning_rate = trial.suggest_float('learning_rate', 0.05, 0.15)
    num_leaves = trial.suggest_int('num_leaves', 20, 40)  
    max_depth = trial.suggest_int('max_depth', 5, 20) 
    min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 1, 10)
    lambda_l1 = trial.suggest_float('lambda_l1', 2, 3)
    lambda_l2 = trial.suggest_float('lambda_l2', 2, 3)

    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,  
        random_state=42,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_data_in_leaf=min_data_in_leaf,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        verbosity=-1  
    )

    # Return the cross-validated mean MSE as the objective value for optimization
    neg_mse_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    mean_mse = -neg_mse_scores.mean()
    return mean_mse

# LightGBM Optimization Techniques Implementations

# Bayesian Optimization using TPE
def bayesian_optimization_tpe():
    print("Starting Bayesian Optimization with TPE...")
    start_time = time.time()
    study = optuna.create_study(sampler=TPESampler(seed=state))
    study.optimize(objective, n_trials=iter)
    best_params = study.best_params
    metrics = evaluate_model(lgb.LGBMRegressor(**best_params, random_state=42))

    # Append best metrics and parameters to CSV
    append_metrics_to_csv('LightGBM_BO_TPE', metrics, time.time() - start_time)
    append_best_params_to_csv('LightGBM_BO_TPE', best_params)

# Bayesian Optimization using Gaussian Process
def bayesian_optimization_gp():
    print("Starting Bayesian Optimization with GP...")
    start_time = time.time()
    study = optuna.create_study(sampler=GPSampler(seed=state))
    study.optimize(objective, n_trials=iter)
    best_params = study.best_params
    metrics = evaluate_model(lgb.LGBMRegressor(**best_params, random_state=42))

    append_metrics_to_csv('LightGBM_BO_GP', metrics, time.time() - start_time)
    append_best_params_to_csv('LightGBM_BO_GP', best_params)
    
    
# Bayesian Optimization using CMAES
def bayesian_optimization_cmaes():
    print("Starting Bayesian Optimization with CMAES...")
    start_time = time.time()
    study = optuna.create_study(sampler=CmaEsSampler(seed=state))
    study.optimize(objective, n_trials=iter)
    best_params = study.best_params
    metrics = evaluate_model(lgb.LGBMRegressor(**best_params, random_state=42), X_train, y_train)

    append_metrics_to_csv('LightGBM_BO_CMAES', metrics, time.time() - start_time)
    append_best_params_to_csv('LightGBM_BO_CMAES', best_params)
    
# Function to run Grid Search
def grid_search_optimization():
    print("Starting Grid Search Optimization...")
    start_time = time.time()
    
    param_grid = {
        'n_estimators': [50, 150],  # Within the range of 50 to 150
        'learning_rate': [0.05, 0.15],  # Within the range of 0.05 to 0.15
        'num_leaves': [20, 40],  # Within the range of 20 to 40
        'max_depth': [5,  20],  # Within the range of 5 to 20
        'min_data_in_leaf': [1, 10],  # Within the range of 1 to 10
        'lambda_l1': [2.0, 3.0],  # Within the range of 2 to 3
        'lambda_l2': [2.0, 3.0]   # Within the range of 2 to 3
    }
    model = lgb.LGBMRegressor(random_state=42, verbosity=-1)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    metrics = evaluate_model(lgb.LGBMRegressor(**best_params, random_state=42))
    
    # Append results to CSV
    append_metrics_to_csv('LightGBM_Grid_Search', metrics, time.time() - start_time)

# Function to run Random Search
def random_search_optimization():
    print("Starting Random Search Optimization...")
    start_time = time.time()
    
    param_distributions = {
        'n_estimators': [50, 100, 150],  # Within the range of 50 to 150
        'learning_rate': [0.05, 0.1, 0.15],  # Within the range of 0.05 to 0.15
        'num_leaves': [20, 30, 40],  # Within the range of 20 to 40
        'max_depth': [5, 10, 15, 20],  # Within the range of 5 to 20
        'min_data_in_leaf': [1, 5, 10],  # Within the range of 1 to 10
        'lambda_l1': [2.0, 2.5, 3.0],  # Within the range of 2 to 3
        'lambda_l2': [2.0, 2.5, 3.0]   # Within the range of 2 to 3
    }
    
    model = lgb.LGBMRegressor(random_state=42, verbosity=-1)
    random_search = RandomizedSearchCV(model, param_distributions, n_iter=50, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
    random_search.fit(X_train, y_train)
    
    best_params = random_search.best_params_
    metrics = evaluate_model(lgb.LGBMRegressor(**best_params, random_state=42))
    
    # Append results to CSV
    append_metrics_to_csv('LightGBM_Random_Search', metrics, time.time() - start_time)

# Main execution
if __name__ == "__main__":
    bayesian_optimization_tpe()
    bayesian_optimization_gp()
    bayesian_optimization_cmaes()
    grid_search_optimization()
    random_search_optimization()

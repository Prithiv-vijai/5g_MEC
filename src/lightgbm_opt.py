import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import os
import time
import optuna
from optuna.samplers import TPESampler, GPSampler, CmaEsSampler, QMCSampler

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
def evaluate_model(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    mean_mse = -scores.mean()
    model.fit(X, y)
    y_pred = model.predict(X)
    return calculate_metrics(y, y_pred)  # Return metrics based on predictions

# Append best parameters to CSV
def append_best_params_to_csv(model_name, best_params):
    for key in best_params:
        if best_params[key] is None:
            best_params[key] = 'None'

    ordered_params = {
        'Model Name': [model_name],
        'num_leaves': [best_params.get('num_leaves', 'None')],
        'learning_rate': [best_params.get('learning_rate', 'None')],
        'max_depth': [best_params.get('max_depth', 'None')],
        'min_data_in_leaf': [best_params.get('min_data_in_leaf', 'None')],
        'lambda_l1': [best_params.get('lambda_l1', 'None')],
        'lambda_l2': [best_params.get('lambda_l2', 'None')],
        'bagging_fraction': [best_params.get('bagging_fraction', 'None')],
        'feature_fraction': [best_params.get('feature_fraction', 'None')],
        'min_split_gain': [best_params.get('min_split_gain', 'None')],
        'max_bin': [best_params.get('max_bin', 'None')]
    }

    df_params = pd.DataFrame(ordered_params)
    file_path = '../data/light_gbm_best_params.csv'

    if not os.path.isfile(file_path):
        df_params.to_csv(file_path, mode='w', header=True, index=False)
    else:
        df_params.to_csv(file_path, mode='a', header=False, index=False)

# Global objective function
def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1)
    num_leaves = trial.suggest_int('num_leaves', 20, 100)
    max_depth = trial.suggest_int('max_depth', 5, 50)
    min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 10, 50)
    lambda_l1 = trial.suggest_float('lambda_l1', 0, 1)
    lambda_l2 = trial.suggest_float('lambda_l2', 0, 1)
    bagging_fraction = trial.suggest_float('bagging_fraction', 0.4, 1.0)
    feature_fraction = trial.suggest_float('feature_fraction', 0.4, 1.0)
    min_split_gain = trial.suggest_float('min_split_gain', 0.0, 1.0)
    max_bin = trial.suggest_int('max_bin', 10, 100)

    model = lgb.LGBMRegressor(
        random_state=42,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_data_in_leaf=min_data_in_leaf,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        bagging_fraction=bagging_fraction,
        feature_fraction=feature_fraction,
        min_split_gain=min_split_gain,
        max_bin=max_bin,
        verbosity=-1  
    )

    return evaluate_model(model, X_train, y_train)[0]

# LightGBM Optimization Techniques Implementations

# Bayesian Optimization using TPE
def bayesian_optimization_tpe():
    print("Starting Bayesian Optimization with TPE...")
    start_time = time.time()
    study = optuna.create_study(sampler=TPESampler(seed=state))
    study.optimize(objective, n_trials=iter)
    best_params = study.best_params
    metrics = evaluate_model(lgb.LGBMRegressor(**best_params, random_state=42), X_train, y_train)

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
    metrics = evaluate_model(lgb.LGBMRegressor(**best_params, random_state=42), X_train, y_train)

    append_metrics_to_csv('LightGBM_BO_GP', metrics, time.time() - start_time)
    append_best_params_to_csv('LightGBM_BO_GP', best_params)

# CMA-ES Sampler
def cmaes_optimization():
    print("Starting CMA-ES Optimization...")
    start_time = time.time()
    study = optuna.create_study(sampler=CmaEsSampler(seed=state))
    study.optimize(objective, n_trials=iter)
    best_params = study.best_params
    metrics = evaluate_model(lgb.LGBMRegressor(**best_params, random_state=42), X_train, y_train)

    append_metrics_to_csv('LightGBM_BO_CMAES', metrics, time.time() - start_time)
    append_best_params_to_csv('LightGBM_BO_CMAES', best_params)

# Quasi-Monte Carlo
def quasi_monte_carlo():
    print("Starting Quasi-Monte Carlo Optimization...")
    start_time = time.time()
    study = optuna.create_study(sampler=QMCSampler(seed=state))
    study.optimize(objective, n_trials=iter)
    best_params = study.best_params
    metrics = evaluate_model(lgb.LGBMRegressor(**best_params, random_state=42), X_train, y_train)

    append_metrics_to_csv('LightGBM_BO_QMC', metrics, time.time() - start_time)
    append_best_params_to_csv('LightGBM_BO_QMC', best_params)

# Running all optimization techniques
bayesian_optimization_tpe()
bayesian_optimization_gp()
cmaes_optimization()
quasi_monte_carlo()

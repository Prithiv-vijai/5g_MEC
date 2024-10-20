import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import os
import time
import optuna
from optuna.samplers import TPESampler, GPSampler ,CmaEsSampler
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
def append_metrics_to_csv(model_name, metrics, completion_time, model_category='Optimization Models'):
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
        'n_estimators': [best_params.get('n_estimators', 'None')],
        'max_depth': [best_params.get('max_depth', 'None')],
        'learning_rate': [best_params.get('learning_rate', 'None')],
        'lambda': [best_params.get('lambda', 'None')],
        'alpha': [best_params.get('alpha', 'None')],
        'gamma': [best_params.get('gamma', 'None')],
        'min_child_weight': [best_params.get('min_child_weight', 'None')],
        'max_leaves': [best_params.get('max_leaves', 'None')],
    }

    df_params = pd.DataFrame(ordered_params)
    file_path = '../data/xgboost_best_params.csv'

    if not os.path.isfile(file_path):
        df_params.to_csv(file_path, mode='w', header=True, index=False)
    else:
        df_params.to_csv(file_path, mode='a', header=False, index=False)

# Global objective function for optimization
def objective_xgb(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 250)
    max_depth = trial.suggest_int('max_depth', 5, 20)
    learning_rate = trial.suggest_float('learning_rate', 0.05, 0.1)
    lambda_ = trial.suggest_float('lambda', 1, 3)
    alpha = trial.suggest_float('alpha', 1, 2)
    gamma = trial.suggest_float('gamma', 1, 2)
    min_child_weight = trial.suggest_int('min_child_weight', 25, 75)  
    max_leaves = trial.suggest_int('max_leaves', 5, 40)              

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        lambda_=lambda_,
        alpha=alpha,
        gamma=gamma,
        min_child_weight=min_child_weight,  # Update to min_child_weight
        max_leaves=max_leaves,
        verbosity=0,
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
    study = optuna.create_study(sampler=TPESampler(seed=state))
    study.optimize(objective_xgb, n_trials=iter)
    best_params = study.best_params
    metrics = evaluate_model(xgb.XGBRegressor(**best_params, random_state=state), X_train, y_train)

    # Append best metrics and parameters to CSV
    append_metrics_to_csv('XGBoost_BO_TPE', metrics, time.time() - start_time)
    append_best_params_to_csv('XGBoost_BO_TPE', best_params)

# Bayesian Optimization using Gaussian Process
def bayesian_optimization_gp():
    print("Starting Bayesian Optimization with GP...")
    start_time = time.time()
    study = optuna.create_study(sampler=GPSampler(seed=state))
    study.optimize(objective_xgb, n_trials=iter)
    best_params = study.best_params
    metrics = evaluate_model(xgb.XGBRegressor(**best_params, random_state=state), X_train, y_train)

    append_metrics_to_csv('XGBoost_BO_GP', metrics, time.time() - start_time)
    append_best_params_to_csv('XGBoost_BO_GP', best_params)
    
    

def bayesian_optimization_cmaes():
    print("Starting Bayesian Optimization with CMAES...")
    start_time = time.time()
    study = optuna.create_study(sampler=CmaEsSampler(seed=state))
    study.optimize(objective_xgb, n_trials=iter)
    best_params = study.best_params
    metrics = evaluate_model(xgb.XGBRegressor(**best_params, random_state=state), X_train, y_train)

    append_metrics_to_csv('XGBoost_BO_CMAES', metrics, time.time() - start_time)
    append_best_params_to_csv('XGBoost_BO_CMAES', best_params)
    
# Grid Search for XGBoost
def grid_search_xgboost():
    print("Starting Grid Search...")
    start_time = time.time()
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [ 10, 15],
        'learning_rate': [ 0.1, 0.15],
        'lambda': [1, 2],
        'alpha': [1, 2],
        'gamma': [0, 0.5],
        'min_child_weight': [25, 50],
        'max_leaves': [10, 20]
    }

    model = xgb.XGBRegressor(random_state=state)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    metrics = calculate_metrics(y_test, grid_search.predict(X_test))

    # Append metrics and parameters to CSV
    append_metrics_to_csv('XGBoost_GridSearch', metrics, time.time() - start_time)
    append_best_params_to_csv('XGBoost_GridSearch', best_params)

# Randomized Search for XGBoost
def randomized_search_xgboost():
    print("Starting Randomized Search...")
    start_time = time.time()
    param_distributions = {
        'n_estimators': [100, 150,170,300],
        'max_depth': [10, 12, 15],
        'learning_rate': [ 0.15, 0.2],
        'lambda': [ 1,2,3],
        'alpha': [1,2,3],
        'gamma': [0.5, 1],
        'min_child_weight':[10, 50, 65, 75],
        'max_leaves': [5,10, 15, 20]
    }

    model = xgb.XGBRegressor(random_state=state)

    randomized_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        scoring='neg_mean_squared_error',
        cv=5,
        n_iter=50,  # Number of parameter settings sampled
        n_jobs=-1,
        random_state=state,
        verbose=1
    )

    randomized_search.fit(X_train, y_train)
    best_params = randomized_search.best_params_
    metrics = calculate_metrics(y_test, randomized_search.predict(X_test))

    # Append metrics and parameters to CSV
    append_metrics_to_csv('XGBoost_RandomizedSearch', metrics, time.time() - start_time)
    append_best_params_to_csv('XGBoost_RandomizedSearch', best_params)

# Running Grid Search and Randomized Search
grid_search_xgboost()
randomized_search_xgboost()

# Running all optimization techniques
bayesian_optimization_tpe()
bayesian_optimization_gp()
bayesian_optimization_cmaes()
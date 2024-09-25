import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.model_selection import cross_val_score
import optuna
from optuna.samplers import TPESampler
import os 
import time  # Import the time module

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
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, rmse, mae, r2, mape

# Function to append metrics to CSV with completion time
def append_metrics_to_csv(model_name, metrics, completion_time, model_category='Boosting Models'):
    # Define the order of columns as per your requirement
    column_order = ['Model Name', 'Model Category', 'MSE', 'RMSE', 'MAE', 'R2', 'MAPE', 'Completion_Time']
    
    # Create a dictionary with correct keys and values
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
    
    # Convert dictionary to DataFrame
    df_metrics = pd.DataFrame(metrics_dict)
    
    # Append DataFrame to CSV
    file_path = '../data/model_performance_metrics.csv'
    if not os.path.isfile(file_path):
        df_metrics.to_csv(file_path, mode='w', header=True, index=False, columns=column_order)
    else:
        df_metrics.to_csv(file_path, mode='a', header=False, index=False, columns=column_order)
        
# Function to save best parameters to CSV
def append_best_params_to_csv(model_name, best_params):
    # Handle missing or None values
    for key in best_params:
        if best_params[key] is None:
            best_params[key] = 'None'
    
    params_dict = {'Model Name': [model_name]}
    params_dict.update({param: [value] for param, value in best_params.items()})
    
    df_params = pd.DataFrame(params_dict)
    file_path = '../data/model_best_params.csv'
    
    if not os.path.isfile(file_path):
        df_params.to_csv(file_path, mode='w', header=True, index=False)
    else:
        df_params.to_csv(file_path, mode='a', header=False, index=False)

# Bayesian Optimization
start_time = time.time()  # Start time
search_space = {
    'learning_rate': Real(0.001, 0.5, prior='uniform'),
    'max_iter': Integer(100, 500),
    'max_leaf_nodes': Integer(20, 100),
    'max_depth': Integer(5, 25),
    'min_samples_leaf': Integer(10, 50),
    'l2_regularization': Real(0, 2, prior='uniform')
}

model = HistGradientBoostingRegressor(random_state=10)
bayes_search = BayesSearchCV(model, search_space, n_iter=50, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
bayes_search.fit(X_train, y_train)

y_pred_bayes = bayes_search.best_estimator_.predict(X_test)
metrics_bayes = calculate_metrics(y_test, y_pred_bayes)

completion_time_bayes = time.time() - start_time  # End time
append_metrics_to_csv('Hgbrt_BO_GP', metrics_bayes, completion_time_bayes)
append_best_params_to_csv('Hgbrt_BO_GP', bayes_search.best_params_)

# Bayesian Optimization (Hyperband)
start_time = time.time()  # Start time
def objective(trial):
    model = HistGradientBoostingRegressor(
        learning_rate=trial.suggest_loguniform('learning_rate', 0.001, 0.5),
        max_iter=trial.suggest_int('max_iter', 100, 500),
        max_leaf_nodes=trial.suggest_int('max_leaf_nodes', 20, 100),
        max_depth=trial.suggest_int('max_depth', 5, 25),
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 10, 50),
        l2_regularization=trial.suggest_uniform('l2_regularization', 0, 2),
        random_state=10
    )
    
    score = -cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error').mean()
    return score

study = optuna.create_study(direction='minimize', sampler=TPESampler())
study.optimize(objective, n_trials=50)

best_params_hyperband = study.best_params
model_hyperband = HistGradientBoostingRegressor(**best_params_hyperband, random_state=10)
model_hyperband.fit(X_train, y_train)

y_pred_hyperband = model_hyperband.predict(X_test)
metrics_hyperband = calculate_metrics(y_test, y_pred_hyperband)

completion_time_hyperband = time.time() - start_time  # End time
append_metrics_to_csv('Hgbrt_BO_HB', metrics_hyperband, completion_time_hyperband)
append_best_params_to_csv('Hgbrt_BO_HB', best_params_hyperband)

# Bayesian Optimization (TPE)
start_time = time.time()

# Define the objective function
def objective(params):
    # Convert the necessary params to int (as required by HistGradientBoostingRegressor)
    params['max_iter'] = int(params['max_iter'])
    params['max_leaf_nodes'] = int(params['max_leaf_nodes'])
    params['max_depth'] = int(params['max_depth'])
    params['min_samples_leaf'] = int(params['min_samples_leaf'])
    
    # Create and evaluate the model
    model = HistGradientBoostingRegressor(**params, random_state=10)
    score = -cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error').mean()
    
    return {'loss': score, 'status': STATUS_OK}

# Define the hyperparameter search space
space = {
    'learning_rate': hp.uniform('learning_rate', 0.001, 0.5),
    'max_iter': hp.quniform('max_iter', 100, 500, 1),
    'max_leaf_nodes': hp.quniform('max_leaf_nodes', 20, 100, 1),
    'max_depth': hp.quniform('max_depth', 5, 25, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 10, 50, 1),
    'l2_regularization': hp.uniform('l2_regularization', 0, 2)
}

# Initialize trials object
trials = Trials()

# Perform optimization using TPE algorithm
best_tpe = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)

# Convert parameters to integer where necessary
best_params_tpe = {k: int(v) if isinstance(v, float) and k in ['max_iter', 'max_leaf_nodes', 'max_depth', 'min_samples_leaf'] else v for k, v in best_tpe.items()}

# Train the final model with the best hyperparameters
model_tpe = HistGradientBoostingRegressor(**best_params_tpe, random_state=10)
model_tpe.fit(X_train, y_train)

# Make predictions
y_pred_tpe = model_tpe.predict(X_test)

# Calculate evaluation metrics
metrics_tpe = calculate_metrics(y_test, y_pred_tpe)

# Calculate the total time taken
completion_time_tpe = time.time() - start_time

# Append the results to CSV files
append_metrics_to_csv('Hgbrt_BO_TPE', metrics_tpe, completion_time_tpe)
append_best_params_to_csv('Hgbrt_BO_TPE', best_params_tpe)

# Grid Search
start_time = time.time()  # Start time
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

completion_time_grid = time.time() - start_time  # End time
append_metrics_to_csv('Hgbrt_Grid', metrics_grid, completion_time_grid)
append_best_params_to_csv('Hgbrt_Grid', grid_search.best_params_)

# Random Search
start_time = time.time()  # Start time
param_grid_random = {
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.5],
    'max_iter': [100, 200, 300, 400, 500],
    'max_leaf_nodes': [20, 31, 50, 70, 100],
    'max_depth': [5, 10, 15, 20, 25],
    'min_samples_leaf': [10, 20, 30, 40, 50],
    'l2_regularization': [0, 0.1, 0.5, 0.8, 1]
}

random_search = RandomizedSearchCV(model, param_distributions=param_grid_random, n_iter=50, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
random_search.fit(X_train, y_train)

y_pred_random = random_search.best_estimator_.predict(X_test)
metrics_random = calculate_metrics(y_test, y_pred_random)

completion_time_random = time.time() - start_time  # End time
append_metrics_to_csv('Hgbrt_Random', metrics_random, completion_time_random)
append_best_params_to_csv('Hgbrt_Random', random_search.best_params_)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import HistGradientBoostingRegressor
import os
import time
import optuna
from optuna.samplers import TPESampler, RandomSampler,GPSampler

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
        'l2_regularization': [best_params.get('l2_regularization', 'None')],
        'learning_rate': [best_params.get('learning_rate', 'None')],
        'max_depth': [best_params.get('max_depth', 'None')],
        'max_iter': [best_params.get('max_iter', 'None')],
        'max_leaf_nodes': [best_params.get('max_leaf_nodes', 'None')],
        'min_samples_leaf': [best_params.get('min_samples_leaf', 'None')]
    }

    df_params = pd.DataFrame(ordered_params)
    file_path = '../data/model_best_params.csv'

    if not os.path.isfile(file_path):
        df_params.to_csv(file_path, mode='w', header=True, index=False)
    else:
        df_params.to_csv(file_path, mode='a', header=False, index=False)

# **Optimization Techniques Implementations**

#  Bayesian Optimization using  TPE**
def bayesian_optimization_tpe():
    print("Starting Bayesian Optimization with TPE...")

    def objective(trial):
        learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1)
        max_iter = trial.suggest_int('max_iter',100, 300)
        max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 5, 50)
        max_depth = trial.suggest_int('max_depth', 5, 25)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 10, 50)
        l2_regularization = trial.suggest_float('l2_regularization', 1, 5)

        model = HistGradientBoostingRegressor(random_state=42,
                                               learning_rate=learning_rate,
                                               max_iter=max_iter,
                                               max_leaf_nodes=max_leaf_nodes,
                                               max_depth=max_depth,
                                               min_samples_leaf=min_samples_leaf,
                                               l2_regularization=l2_regularization)

        return evaluate_model(model, X_train, y_train)[0]  # Use K-fold CV here

    start_time = time.time()
    study = optuna.create_study(sampler=TPESampler(seed=state))
    study.optimize(objective, n_trials=iter)
    best_params = study.best_params
    metrics = evaluate_model(HistGradientBoostingRegressor(**best_params, random_state=42), X_train, y_train)
    
    # Append best metrics and parameters to CSV
    append_metrics_to_csv('Hgbrt_BO_TPE', metrics, time.time() - start_time)
    append_best_params_to_csv('Hgbrt_BO_TPE', best_params)

# Bayesian Optimization using Gaussian Process
def bayesian_optimization_gp():
    print("Starting Bayesian Optimization with GP...")

    def objective(trial):
        learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1)
        max_iter = trial.suggest_int('max_iter',100, 300)
        max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 5, 50)
        max_depth = trial.suggest_int('max_depth', 5, 25)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 10, 50)
        l2_regularization = trial.suggest_float('l2_regularization', 1, 5)

        model = HistGradientBoostingRegressor(random_state=42,
                                               learning_rate=learning_rate,
                                               max_iter=max_iter,
                                               max_leaf_nodes=max_leaf_nodes,
                                               max_depth=max_depth,
                                               min_samples_leaf=min_samples_leaf,
                                               l2_regularization=l2_regularization)

        return evaluate_model(model, X_train, y_train)[0]  # Use K-fold CV here

    start_time = time.time()
    study = optuna.create_study(sampler=optuna.samplers.GPSampler(seed=state))
    study.optimize(objective, n_trials=iter)
    best_params = study.best_params
    metrics = evaluate_model(HistGradientBoostingRegressor(**best_params, random_state=42), X_train, y_train)
    
    # Append best metrics and parameters to CSV
    append_metrics_to_csv('Hgbrt_BO_GP', metrics, time.time() - start_time)
    append_best_params_to_csv('Hgbrt_BO_GP', best_params)


    
# **CMA-ES Sampler**
def cmaes_optimization():
    print("Starting CMA-ES Optimization...")

    def objective(trial):
        learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1)
        max_iter = trial.suggest_int('max_iter',100, 300)
        max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 5, 50)
        max_depth = trial.suggest_int('max_depth', 5, 25)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 10, 50)
        l2_regularization = trial.suggest_float('l2_regularization', 1, 5)

        model = HistGradientBoostingRegressor(random_state=42,
                                               learning_rate=learning_rate,
                                               max_iter=max_iter,
                                               max_leaf_nodes=max_leaf_nodes,
                                               max_depth=max_depth,
                                               min_samples_leaf=min_samples_leaf,
                                               l2_regularization=l2_regularization)

        return evaluate_model(model, X_train, y_train)[0]  # Use K-fold CV here

    start_time = time.time()
    study = optuna.create_study(sampler=optuna.samplers.CmaEsSampler(seed=state))
    study.optimize(objective, n_trials=iter)
    best_params = study.best_params
    metrics = evaluate_model(HistGradientBoostingRegressor(**best_params, random_state=42), X_train, y_train)
    
    append_metrics_to_csv('Hgbrt_CMAES', metrics, time.time() - start_time)
    append_best_params_to_csv('Hgbrt_CMAES', best_params)


# **QMC Sampler**
def qmc_optimization():
    print("Starting QMC Optimization...")

    def objective(trial):
        learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1)
        max_iter = trial.suggest_int('max_iter',100, 300)
        max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 5, 50)
        max_depth = trial.suggest_int('max_depth', 5, 25)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 10, 50)
        l2_regularization = trial.suggest_float('l2_regularization', 1, 5)

        model = HistGradientBoostingRegressor(random_state=42,
                                               learning_rate=learning_rate,
                                               max_iter=max_iter,
                                               max_leaf_nodes=max_leaf_nodes,
                                               max_depth=max_depth,
                                               min_samples_leaf=min_samples_leaf,
                                               l2_regularization=l2_regularization)

        return evaluate_model(model, X_train, y_train)[0]  # Use K-fold CV here

    start_time = time.time()
    study = optuna.create_study(sampler=optuna.samplers.QMCSampler(seed=state))
    study.optimize(objective, n_trials=iter)
    best_params = study.best_params
    metrics = evaluate_model(HistGradientBoostingRegressor(**best_params, random_state=42), X_train, y_train)
    
    append_metrics_to_csv('Hgbrt_QMC', metrics, time.time() - start_time)
    append_best_params_to_csv('Hgbrt_QMC', best_params)

# Run the optimization functions
if __name__ == '__main__':
    bayesian_optimization_tpe()
    bayesian_optimization_gp()
    cmaes_optimization()
    qmc_optimization()


import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
import lightgbm as lgb
import os

# Load the dataset from a CSV file
data = pd.read_csv("../data/augmented_dataset.csv")

# Define features and target
X = data[['Application_Type', 'Signal_Strength', 'Latency', 'Required_Bandwidth', 'Allocated_Bandwidth']]
y = data['Resource_Allocation']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Function to calculate metrics
def calculate_metrics(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, rmse, mae, r2, mape

# Function to append metrics to CSV
def append_metrics_to_csv(model_name, metrics, completion_time, model_category='Optimization Models'):
    column_order = ['Model Name', 'Model Category', 'MSE', 'RMSE', 'MAE', 'R2', 'MAPE', 'Completion Time (s)']
    metrics_dict = {
        'Model Name': [model_name],
        'Model Category': [model_category],
        'MSE': [metrics[0]],
        'RMSE': [metrics[1]],
        'MAE': [metrics[2]],
        'R2': [metrics[3]],
        'MAPE': [metrics[4]],
        'Completion Time (s)': [completion_time]
    }
    df_metrics = pd.DataFrame(metrics_dict)
    file_path = "../data/model_performance_metrics.csv"
    if not os.path.isfile(file_path):
        df_metrics.to_csv(file_path, mode='w', header=True, index=False, columns=column_order)
    else:
        df_metrics.to_csv(file_path, mode='a', header=False, index=False, columns=column_order)

# Function to append best parameters to CSV
def append_best_params_to_csv(model_name, best_params):
    for key in best_params:
        if best_params[key] is None:
            best_params[key] = 'None'

    ordered_params = {
        'Model Name': [model_name],
        'num_leaves': [best_params.get('num_leaves', 'None')],
        'n_estimators': [best_params.get('n_estimators', 'None')],  # Added n_estimators
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

# Define the bounds for SMA optimization for LightGBM
bounds = np.array([
    [100, 250],      # n_estimators
    [0.05, 0.09],    # learning_rate
    [5, 40],         # num_leaves
    [5, 20],         # max_depth
    [35, 75],        # min_data_in_leaf
    [1, 3],          # lambda_l1
    [1, 3]           # lambda_l2
])

# Helper function to convert an array to a dictionary for model parameters
def array_to_dict(agent):
    return {
        'n_estimators': int(agent[0]),
        'learning_rate': agent[1],
        'num_leaves': int(agent[2]),
        'max_depth': int(agent[3]),
        'min_data_in_leaf': int(agent[4]),
        'lambda_l1': agent[5],
        'lambda_l2': agent[6]
    }

# Slime Mould Algorithm (SMA)
def initialize_population(n_agents, dim, bounds):
    population = np.random.rand(n_agents, dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    return population

def fitness_function(model, X_train, y_train, X_test, y_test, params):
    model.set_params(**params)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
    return -np.mean(scores)

def slime_mould_algorithm(model, X_train, y_train, X_test, y_test, bounds, n_agents, max_iter):
    dim = bounds.shape[0]
    population = initialize_population(n_agents, dim, bounds)
    fitness = np.zeros(n_agents)
    best_agent = None
    best_fitness = float('inf')

    for agent in range(n_agents):
        params = array_to_dict(population[agent])
        fitness[agent] = fitness_function(model, X_train, y_train, X_test, y_test, params)
        
        if fitness[agent] < best_fitness:
            best_fitness = fitness[agent]
            best_agent = population[agent].copy()

    for t in range(max_iter):
        print(f"Iteration {t + 1}/{max_iter}")
        for agent in range(n_agents):
            population[agent] = update_position(population[agent], best_agent, fitness, t, max_iter, bounds)
            params = array_to_dict(population[agent])
            fitness[agent] = fitness_function(model, X_train, y_train, X_test, y_test, params)
            
            if fitness[agent] < best_fitness:
                best_fitness = fitness[agent]
                best_agent = population[agent].copy()
        
    return best_agent, best_fitness

def update_position(agent, best_agent, fitness, t, max_iter, bounds):
    b = 1 - t / max_iter
    d = 2 * (np.random.rand() - 0.5)
    new_agent = agent + d * b * (best_agent - agent)

    new_agent = np.clip(new_agent, bounds[:, 0], bounds[:, 1])
    return new_agent

# Define the LightGBM model
model = lgb.LGBMRegressor(random_state=10,verbosity=-1)

# Record the start time of SMA optimization
start_time = time.time()

# Run the SMA optimization
best_agent, best_fitness = slime_mould_algorithm(model, X_train, y_train, X_test, y_test, bounds, n_agents=20, max_iter=200)

# Calculate completion time for SMA optimization
sma_completion_time = time.time() - start_time

# Use the best parameters found by SMA to train the final model
best_params = array_to_dict(best_agent)
model.set_params(**best_params)
model.fit(X_train, y_train)

# Predict and calculate metrics
y_pred_sma = model.predict(X_train)
metrics_sma = calculate_metrics(y_train, y_pred_sma)

# Append SMA results with completion time to CSV
append_metrics_to_csv('LightGBM_SMA', metrics_sma, sma_completion_time)

# Append the best parameters to CSV
append_best_params_to_csv('LightGBM_SMA', best_params)

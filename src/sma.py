import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
import os

# Load the dataset from a CSV file
data = pd.read_csv("../data/augmented_datasett.csv")

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

# Function to append metrics to CSV
def append_metrics_to_csv(model_name, metrics, model_category='Boosting Models'):
    column_order = ['Model Name', 'Model Category', 'MSE', 'RMSE', 'MAE', 'R2', 'MAPE']
    metrics_dict = {
        'Model Name': [model_name],
        'Model Category': [model_category],
        'MSE': [metrics[0]],
        'RMSE': [metrics[1]],
        'MAE': [metrics[2]],
        'R2': [metrics[3]],
        'MAPE': [metrics[4]],
    }
    df_metrics = pd.DataFrame(metrics_dict)
    file_path = "../data/model_performance_metrics.csv"
    if not os.path.isfile(file_path):
        df_metrics.to_csv(file_path, mode='w', header=True, index=False, columns=column_order)
    else:
        df_metrics.to_csv(file_path, mode='a', header=False, index=False, columns=column_order)

# Function to append best parameters to CSV
def append_best_params_to_csv(model_name, best_params):
    df_params = pd.DataFrame([best_params])
    df_params.insert(0, 'Model Name', model_name)
    
    file_path = "../data/model_best_params.csv"
    if not os.path.isfile(file_path):
        df_params.to_csv(file_path, mode='w', header=True, index=False)
    else:
        df_params.to_csv(file_path, mode='a', header=False, index=False)

# Define the bounds for SMA optimization
bounds = np.array([
    [0.001, 0.5],    # learning_rate
    [100, 500],      # max_iter
    [20, 100],       # max_leaf_nodes
    [5, 25],         # max_depth
    [10, 50],        # min_samples_leaf
    [0, 2]           # l2_regularization
])

# Helper function to convert an array to a dictionary for model parameters
def array_to_dict(agent):
    return {
        'learning_rate': agent[0],
        'max_iter': int(agent[1]),
        'max_leaf_nodes': int(agent[2]),
        'max_depth': int(agent[3]),
        'min_samples_leaf': int(agent[4]),
        'l2_regularization': agent[5]
    }

# Objective function to calculate Negative MSE (NMSE)
def objective_function(params, model, X_train, y_train):
    model.set_params(**params)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    return -np.mean(scores)  # Negative MSE for minimization

# Slime Mould Algorithm (SMA)
def initialize_population(n_agents, dim, bounds):
    population = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_agents, dim))
    return population

def slime_mould_algorithm(model, X_train, y_train, X_test, y_test, bounds, n_agents=50, max_iter=100):
    dim = bounds.shape[0]
    positions = initialize_population(n_agents, dim, bounds)
    fitness = np.zeros(n_agents)
    best_agent = None
    best_fitness = float('inf')
    w = 0.9
    z = 0.1
    vb = 0

    for i in range(max_iter):
        # Calculate fitness for each agent
        for agent in range(n_agents):
            params = array_to_dict(positions[agent])
            fitness[agent] = objective_function(params, model, X_train, y_train)
            if fitness[agent] < best_fitness:
                best_fitness = fitness[agent]
                best_agent = positions[agent].copy()

        # Sort agents by fitness
        sorted_indices = np.argsort(fitness)
        positions = positions[sorted_indices]

        # Update positions
        w *= np.exp(-i / max_iter)
        for j in range(n_agents):
            if random.random() < w:
                best_pos = positions[0]
                positions[j] += np.random.rand() * (best_pos - positions[j])
            else:
                random_index = random.randint(0, n_agents - 1)
                random_pos = positions[random_index]
                positions[j] += z * np.random.rand() * (random_pos - positions[j])
            
            # Add random noise based on velocity
            positions[j] += vb * np.random.randn(dim)
            # Ensure the new position is within bounds
            positions[j] = np.clip(positions[j], bounds[:, 0], bounds[:, 1])

    return best_agent, best_fitness

# Define the HGBRT model
model = HistGradientBoostingRegressor(random_state=42)

# Run the SMA optimization
best_agent, best_fitness = slime_mould_algorithm(model, X_train, y_train, X_test, y_test, bounds, n_agents=100, max_iter=200)

# Use the best parameters found by SMA to train the final model
best_params = array_to_dict(best_agent)
model.set_params(**best_params)
model.fit(X_train, y_train)

# Predict and calculate metrics
y_pred_sma = model.predict(X_test)
metrics_sma = calculate_metrics(y_test, y_pred_sma)

# Append SMA results to CSV
append_metrics_to_csv('Hgbrt_sma_new', metrics_sma)

# Append the best parameters to CSV
append_best_params_to_csv('Hgbrt_sma_new', best_params)
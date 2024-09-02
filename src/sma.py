import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
    # Define the order of columns as per your requirement
    column_order = ['Model Name', 'Model Category', 'MSE', 'RMSE', 'MAE', 'R2', 'MAPE']
    
    # Create a dictionary with correct keys and values
    metrics_dict = {
        'Model Name': [model_name],
        'Model Category': [model_category],
        'MSE': [metrics[0]],
        'RMSE': [metrics[1]],
        'MAE': [metrics[2]],
        'R2': [metrics[3]],
        'MAPE': [metrics[4]],
    }
    
    # Convert dictionary to DataFrame
    df_metrics = pd.DataFrame(metrics_dict)
    
    # Append DataFrame to CSV
    file_path = "../data/model_performance_metrics.csv"
    if not os.path.isfile(file_path):
        df_metrics.to_csv(file_path, mode='w', header=True, index=False, columns=column_order)
    else:
        df_metrics.to_csv(file_path, mode='a', header=False, index=False, columns=column_order)

# Define the bounds for SMA optimization
# Define the bounds for SMA optimization
bounds = np.array([
    [0.001, 0.2],    # learning_rate: Generally smaller learning rates are better for gradient boosting.
    [100, 1000],     # max_iter: Allow for a higher number of iterations to ensure model convergence.
    [10, 500],       # max_leaf_nodes: More flexibility in the number of leaf nodes for different levels of model complexity.
    [3, 30],         # max_depth: Ranging depth to control model overfitting.
    [1, 50],         # min_samples_leaf: Control over minimum samples in a leaf node to avoid overfitting.
    [0, 10]          # l2_regularization: Higher range for L2 regularization to test the effects on model performance.
])


# Slime Mould Algorithm (SMA)
def initialize_population(n_agents, dim, bounds):
    population = np.random.rand(n_agents, dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    return population

def fitness_function(model, X_train, y_train, X_test, y_test, params):
    model.set_params(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

def slime_mould_algorithm(model, X_train, y_train, X_test, y_test, bounds, n_agents=30, max_iter=100):
    dim = bounds.shape[0]
    population = initialize_population(n_agents, dim, bounds)
    fitness = np.zeros(n_agents)
    best_agent = np.zeros(dim)
    best_fitness = float('inf')

    for agent in range(n_agents):
        params = {
            'learning_rate': population[agent][0],
            'max_iter': int(population[agent][1]),
            'max_leaf_nodes': int(population[agent][2]),
            'max_depth': int(population[agent][3]),
            'min_samples_leaf': int(population[agent][4]),
            'l2_regularization': population[agent][5]
        }
        fitness[agent] = fitness_function(model, X_train, y_train, X_test, y_test, params)
        
        if fitness[agent] < best_fitness:
            best_fitness = fitness[agent]
            best_agent = population[agent]

    for t in range(max_iter):
        for agent in range(n_agents):
            # Update positions based on SMA rules
            population[agent] = update_position(population[agent], best_agent, fitness, t, max_iter, bounds)
            params = {
                'learning_rate': population[agent][0],
                'max_iter': int(population[agent][1]),
                'max_leaf_nodes': int(population[agent][2]),
                'max_depth': int(population[agent][3]),
                'min_samples_leaf': int(population[agent][4]),
                'l2_regularization': population[agent][5]
            }
            fitness[agent] = fitness_function(model, X_train, y_train, X_test, y_test, params)
            
            if fitness[agent] < best_fitness:
                best_fitness = fitness[agent]
                best_agent = population[agent]
        
    return best_agent, best_fitness

def update_position(agent, best_agent, fitness, t, max_iter, bounds):
    # SMA-specific position update rules go here (simplified example)
    b = 1 - t / max_iter
    d = 2 * (np.random.rand() - 0.5)
    new_agent = agent + d * b * (best_agent - agent)

    # Ensure the new position is within bounds
    new_agent = np.clip(new_agent, bounds[:, 0], bounds[:, 1])
    return new_agent

# Define the HGBRT model
model = HistGradientBoostingRegressor()

# Run the SMA optimization
best_agent, best_fitness = slime_mould_algorithm(model, X_train, y_train, X_test, y_test, bounds, n_agents=30, max_iter=100)

# Use the best parameters found by SMA to train the final model
best_params = {
    'learning_rate': best_agent[0],
    'max_iter': int(best_agent[1]),
    'max_leaf_nodes': int(best_agent[2]),
    'max_depth': int(best_agent[3]),
    'min_samples_leaf': int(best_agent[4]),
    'l2_regularization': best_agent[5]
}
model.set_params(**best_params)
model.fit(X_train, y_train)

# Predict and calculate metrics
y_pred_sma = model.predict(X_test)
metrics_sma = calculate_metrics(y_test, y_pred_sma)

# Append SMA results to CSV
append_metrics_to_csv('Hgbrt_sma', metrics_sma)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score, 
    mean_absolute_percentage_error
)
from sklearn.ensemble import HistGradientBoostingRegressor
import os
import time

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

# Append best parameters to CSV
def append_best_params_to_csv(model_name, best_params):
    ordered_params = {
        'Model Name': [model_name],
        'l2_regularization': [best_params[5]],
        'learning_rate': [best_params[0]],
        'max_depth': [best_params[3]],
        'max_iter': [best_params[1]],
        'max_leaf_nodes': [best_params[2]],
        'min_samples_leaf': [best_params[4]]
    }
    df_params = pd.DataFrame(ordered_params)
    file_path = '../data/model_best_params.csv'
    if not os.path.isfile(file_path):
        df_params.to_csv(file_path, mode='w', header=True, index=False)
    else:
        df_params.to_csv(file_path, mode='a', header=False, index=False)

# Initialize the model
model = HistGradientBoostingRegressor(random_state=40)

def evaluate_model(model, X, y):
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    mse_scores = -cv_scores  # Convert back to positive MSE
    mean_mse = np.mean(mse_scores)
    std_mse = np.std(mse_scores)

    # Train the model and predict on the test set
    model.fit(X, y)
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    return metrics, mean_mse, std_mse

def objective_function(params):
    # Unpack parameters and ensure parameter constraints
    learning_rate = np.clip(params[0], 0.001, 0.5)
    max_iter = np.clip(int(params[1]), 5, 500)  # Already converted to int here
    max_leaf_nodes = max(2, int(params[2]))      # Already converted to int here
    max_depth = max(1, int(params[3]))            # Already converted to int here
    min_samples_leaf = max(1, int(params[4]))     # Already converted to int here
    l2_regularization = np.clip(params[5], 0, 2)

    # Set parameters for the model
    model.set_params(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization
    )

    # Evaluate the model and return the score (mean MSE from CV)
    _, mean_mse, _ = evaluate_model(model, X_train, y_train)
    return mean_mse  # Return the score for minimization

# Define the bounds for the parameters
bounds = [
    (0.001, 0.5),    # learning_rate
    (5, 500),        # max_iter
    (2, 100),        # max_leaf_nodes (minimum 2)
    (1, 25),         # max_depth (minimum 1)
    (1, 50),         # min_samples_leaf (minimum 1)
    (0, 2)           # l2_regularization
]

# Quantum-Inspired Particle Swarm Optimization (QIPSO)
class QIPSO:
    def __init__(self, objective_function, num_particles=30, max_iter=50):
        self.objective_function = objective_function
        self.num_particles = num_particles
        self.max_iter = max_iter
        # Ensure integer values for max_iter, max_depth, max_leaf_nodes, and min_samples_leaf
        self.particles = np.zeros((num_particles, 6))
        self.particles[:, 0] = np.random.uniform(0.001, 0.5, num_particles)  # learning_rate
        self.particles[:, 1] = np.random.randint(5, 501, num_particles)       # max_iter
        self.particles[:, 2] = np.random.randint(2, 101, num_particles)       # max_leaf_nodes
        self.particles[:, 3] = np.random.randint(1, 26, num_particles)        # max_depth
        self.particles[:, 4] = np.random.randint(1, 51, num_particles)        # min_samples_leaf
        self.particles[:, 5] = np.random.uniform(0, 2, num_particles)         # l2_regularization
        self.velocities = np.random.rand(num_particles, 6) * 0.1
        self.best_positions = np.copy(self.particles)
        self.best_scores = np.array([float('inf')] * num_particles)
        self.global_best_position = np.zeros(6)
        self.global_best_score = float('inf')

    def optimize(self):
        for iteration in range(self.max_iter):
            for i in range(self.num_particles):
                # Evaluate the objective function
                score = self.objective_function(self.particles[i])

                # Update the best score and position
                if score < self.best_scores[i]:
                    self.best_scores[i] = score
                    self.best_positions[i] = self.particles[i]

                    if score < self.global_best_score:
                        self.global_best_score = score
                        self.global_best_position = self.particles[i]

            # Update velocities and positions
            for i in range(self.num_particles):
                # Quantum-inspired update
                self.velocities[i] = (
                    0.5 * self.velocities[i] + 
                    2 * np.random.rand() * (self.best_positions[i] - self.particles[i]) + 
                    2 * np.random.rand() * (self.global_best_position - self.particles[i])
                )

                self.particles[i] += self.velocities[i]
                # Ensure particles stay within bounds
                for j in range(len(bounds)):
                    self.particles[i, j] = np.clip(self.particles[i, j], bounds[j][0], bounds[j][1])

            # Print progress for each iteration
            print(f"Iteration {iteration + 1}/{self.max_iter}: Best Score = {self.global_best_score}")

        return self.global_best_position

# Start the QIPSO optimization
start_time_qipso = time.time()
print("Starting Quantum-Inspired Particle Swarm Optimization (QIPSO)...")

qipso_optimizer = QIPSO(objective_function)
best_params_qipso = qipso_optimizer.optimize()

# Use best parameters to make predictions
model.set_params(
    learning_rate=best_params_qipso[0],
    max_iter=int(best_params_qipso[1]),
    max_leaf_nodes=int(best_params_qipso[2]),
    max_depth=int(best_params_qipso[3]),
    min_samples_leaf=int(best_params_qipso[4]),
    l2_regularization=best_params_qipso[5]
)

# Evaluate the model using the best parameters
metrics, mean_mse, std_mse = evaluate_model(model, X_train, y_train)
completion_time = time.time() - start_time_qipso

# Save metrics and best parameters
append_metrics_to_csv('HistGradientBoostingRegressor', metrics, completion_time)
append_best_params_to_csv('HistGradientBoostingRegressor', best_params_qipso)

print("Optimization and evaluation complete.")

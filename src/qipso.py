import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import lightgbm as lgb
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import time

# Load the dataset from a CSV file
data = pd.read_csv('../data/augmented_dataset.csv')

# Define features and target
X = data[['Application_Type', 'Signal_Strength', 'Latency', 'Required_Bandwidth', 'Allocated_Bandwidth']]
y = data['Resource_Allocation']

kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Function to calculate metrics
def calculate_metrics(model, X_train, X_test, y_train, y_test):
    # Train the model
    model.fit(X_train, y_train)
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

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

# Initialize the model
model = lgb.LGBMRegressor(random_state=42, verbose=-1)

# Quantum-Inspired Particle Swarm Optimization (QIPSO)
class QIPSO:
    def __init__(self, objective_function, num_particles=20, max_iter=200):
        self.objective_function = objective_function
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.particles = np.zeros((num_particles, 7))
        self.particles[:, 0] = np.random.uniform(0.05, 0.09, num_particles)  # learning_rate
        self.particles[:, 1] = np.random.randint(100, 250, num_particles)    # n_estimators
        self.particles[:, 2] = np.random.randint(5, 40, num_particles)       # num_leaves
        self.particles[:, 3] = np.random.randint(5, 20, num_particles)       # max_depth
        self.particles[:, 4] = np.random.randint(35, 75, num_particles)      # min_data_in_leaf
        self.particles[:, 5] = np.random.uniform(2, 4, num_particles)        # lambda_l1
        self.particles[:, 6] = np.random.uniform(2, 4, num_particles)        # lambda_l2
        self.velocities = np.random.rand(num_particles, 7) * 0.1  # Changed from 8 to 7
        self.best_positions = np.copy(self.particles)
        self.best_scores = np.array([float('inf')] * num_particles)
        self.global_best_position = np.zeros(7)  # Changed from 8 to 7
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
                self.velocities[i] = (
                    0.5 * self.velocities[i] + 
                    2 * np.random.rand() * (self.best_positions[i] - self.particles[i]) + 
                    2 * np.random.rand() * (self.global_best_position - self.particles[i])
                )
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], bounds[:, 0], bounds[:, 1])

            print(f"Iteration {iteration + 1}/{self.max_iter}: Best Score = {self.global_best_score}")

        return self.global_best_position

# Objective function for optimization
def objective_function(params):
    learning_rate = np.clip(params[0], bounds[0][0], bounds[0][1])
    n_estimators = int(np.clip(params[1], bounds[1][0], bounds[1][1]))
    num_leaves = int(np.clip(params[2], bounds[2][0], bounds[2][1]))
    max_depth = int(np.clip(params[3], bounds[3][0], bounds[3][1]))
    min_data_in_leaf = int(np.clip(params[4], bounds[4][0], bounds[4][1]))
    lambda_l1 = np.clip(params[5], bounds[5][0], bounds[5][1])
    lambda_l2 = np.clip(params[6], bounds[6][0], bounds[6][1])

    model.set_params(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_data_in_leaf,
        reg_lambda=lambda_l2,
        reg_alpha=lambda_l1
    )

    # Perform 5-fold cross-validation
    scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    return np.mean(scores)

# Define parameter bounds
bounds = np.array([
    [0.05, 0.09],   # learning_rate
    [100, 250],    # n_estimators
    [5, 40],        # num_leaves
    [5, 20],        # max_depth
    [35, 75],       # min_data_in_leaf
    [2, 3],         # lambda_l1
    [2, 3]          # lambda_l2
])

# Start the QIPSO optimization
start_time_qipso = time.time()
print("Starting Quantum-Inspired Particle Swarm Optimization (QIPSO)...")

qipso_optimizer = QIPSO(objective_function)
best_params_qipso = qipso_optimizer.optimize()

# Use best parameters to make predictions
model.set_params(
    learning_rate=best_params_qipso[0],
    n_estimators=int(best_params_qipso[1]),
    num_leaves=int(best_params_qipso[2]),
    max_depth=int(best_params_qipso[3]),
    min_child_samples=int(best_params_qipso[4]),
    reg_lambda=best_params_qipso[6],
    reg_alpha=best_params_qipso[5],
    verbose=-1
)

# Train and evaluate the model with the best parameters
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
metrics_qipso = calculate_metrics(model, X_train, X_test, y_train, y_test)

completion_time_qipso = time.time() - start_time_qipso
print(f"Completion Time: {completion_time_qipso:.4f} seconds")
append_metrics_to_csv('LightGBM_QIPSO', metrics_qipso, completion_time_qipso)
append_best_params_to_csv('LightGBM_QIPSO', {
    'n_estimators': best_params_qipso[1],
    'learning_rate': best_params_qipso[0],
    'num_leaves': best_params_qipso[2],
    'max_depth': best_params_qipso[3],
    'min_data_in_leaf': best_params_qipso[4],
    'lambda_l1': best_params_qipso[5],
    'lambda_l2': best_params_qipso[6]
})

print("Optimization and evaluation complete.")

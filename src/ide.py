import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split ,cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import os
import time

# Set a seed for reproducibility
seed = 42
np.random.seed(seed)

# Load the dataset from a CSV file
data = pd.read_csv('../data/augmented_dataset.csv')

# Define features and target
X = data[['Application_Type', 'Signal_Strength', 'Latency', 'Required_Bandwidth', 'Allocated_Bandwidth']]
y = data['Resource_Allocation']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

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
        'lambda_l2': [best_params.get('lambda_l2', 'None')]
    }

    df_params = pd.DataFrame(ordered_params)
    file_path = '../data/light_gbm_best_params.csv'

    if not os.path.isfile(file_path):
        df_params.to_csv(file_path, mode='w', header=True, index=False)
    else:
        df_params.to_csv(file_path, mode='a', header=False, index=False)

# Initialize LightGBM model
model = lgb.LGBMRegressor(random_state=seed,verbose=-1)

def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Train the model and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return calculate_metrics(y_test, y_pred)

# Define the bounds for the LightGBM parameters
bounds = [
    (0.05, 0.09),    # learning_rate
    (100, 250),      # n_estimators
    (5, 40),         # num_leaves (minimum 5)
    (5, 20),         # max_depth (minimum 5)
    (35, 75),        # min_data_in_leaf (minimum 25)
    (2, 4),          # lambda_l1
    (2, 4)           # lambda_l2
]

# Improved Differential Evolution (IDE)
start_time_ide = time.time()
print("Starting Improved Differential Evolution (IDE)...")

# Initialize global best variables
best_global_score = float('inf')
best_global_position = np.zeros(len(bounds))

# Set up parameters for IDE
num_iterations = 100
popsize = 15
mutation_factor = 0.5
recombination_probability = 0.7

# Initialize a population randomly with seed
population = np.random.rand(popsize, len(bounds))
for i in range(len(bounds)):
    population[:, i] = population[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]

best_scores = np.array([float('inf')] * popsize)
best_positions = np.copy(population)

for iteration in range(num_iterations):
    print(f"IDE Iteration {iteration + 1}/{num_iterations}")

    for i in range(popsize):
        indices = np.random.choice(np.delete(np.arange(popsize), i), 3, replace=False)
        a, b, c = population[indices]

        # Mutation
        mutated_vector = np.clip(a + mutation_factor * (b - c), 
                                 [bounds[j][0] for j in range(len(bounds))], 
                                 [bounds[j][1] for j in range(len(bounds))])

        # Crossover
        trial_vector = np.where(np.random.rand(len(bounds)) < recombination_probability, 
                                mutated_vector, population[i])

        # Evaluate the trial vector with updated parameters
        model.set_params(
            learning_rate=trial_vector[0],
            n_estimators=int(trial_vector[1]),
            num_leaves=int(trial_vector[2]),
            max_depth=int(trial_vector[3]),
            min_data_in_leaf=int(trial_vector[4]),
            lambda_l1=trial_vector[5],
            lambda_l2=trial_vector[6]
        )
        
        # Return the cross-validated mean MSE as the objective value for optimization
        neg_mse_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
        mean_mse = -neg_mse_scores.mean()
        trial_score= mean_mse

        # Selection
        if trial_score < best_scores[i]:
            best_scores[i] = trial_score
            best_positions[i] = trial_vector

            # Update global best
            if trial_score < best_global_score:
                best_global_score = trial_score
                best_global_position = trial_vector

# Extract best parameters from the result and convert to a dictionary
best_params_ide = {
    'learning_rate': best_global_position[0],
    'n_estimators': int(best_global_position[1]),
    'num_leaves': int(best_global_position[2]),
    'max_depth': int(best_global_position[3]),
    'min_data_in_leaf': int(best_global_position[4]),
    'lambda_l1': best_global_position[5],
    'lambda_l2': best_global_position[6]
}

# Use best parameters to make predictions
model.set_params(**best_params_ide)

# Fit the model and predict
y_pred_ide = model.fit(X_train, y_train).predict(X_train)
metrics_ide = calculate_metrics(y_train, y_pred_ide)

completion_time_ide = time.time() - start_time_ide
append_metrics_to_csv('LightGBM_IDE', metrics_ide, completion_time_ide)
append_best_params_to_csv('LightGBM_IDE', best_params_ide)

print("Completed Improved Differential Evolution (IDE) with best parameters:", best_params_ide)

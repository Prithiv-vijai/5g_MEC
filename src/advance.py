import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    mean_absolute_error,
)
import optuna
import os
import time

# Set a random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)
optuna.logging.set_verbosity(optuna.logging.WARNING)  # To suppress Optuna's logs

# Load the dataset from a CSV file
data = pd.read_csv('../data/augmented_dataset.csv')

# Define features and target
X = data[['Application_Type', 'Signal_Strength', 'Latency', 'Required_Bandwidth', 'Allocated_Bandwidth']]
y = data['Resource_Allocation']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

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
        'Completion_Time': [completion_time],
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
        'l2_regularization': [best_params.get('l2_regularization', 'None')],
        'learning_rate': [best_params.get('learning_rate', 'None')],
        'max_depth': [best_params.get('max_depth', 'None')],
        'max_iter': [best_params.get('max_iter', 'None')],
        'max_leaf_nodes': [best_params.get('max_leaf_nodes', 'None')],
        'min_samples_leaf': [best_params.get('min_samples_leaf', 'None')],
    }

    df_params = pd.DataFrame(ordered_params)
    file_path = '../data/model_best_params.csv'

    if not os.path.isfile(file_path):
        df_params.to_csv(file_path, mode='w', header=True, index=False)
    else:
        df_params.to_csv(file_path, mode='a', header=False, index=False)




# Define the Population-Based Training (PBT) Setup
class PBT:
    def __init__(self, n_generations=10, n_population=10):
        self.n_generations = n_generations
        self.n_population = n_population
        self.population = self.init_population()
        self.best_model_params = None
        self.best_score = float('inf')

    def init_population(self):
        return [self.sample_model() for _ in range(self.n_population)]

    def sample_model(self):
        return {
            "learning_rate": np.random.uniform(0.001, 0.1),
            "max_iter": np.random.randint(100, 300),
            "max_leaf_nodes": np.random.randint(5, 50),
            "max_depth": np.random.randint(5, 25),
            "min_samples_leaf": np.random.randint(10, 50),
            "l2_regularization": np.random.uniform(1, 5),
        }

    def optimize(self):
        for generation in range(self.n_generations):
            print(f"Generation {generation + 1}/{self.n_generations}...")
            for idx, model_params in enumerate(self.population):
                model = HistGradientBoostingRegressor(**{k: v for k, v in model_params.items() if k != 'score'}, random_state=random_seed)
                score = -cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error').mean()
                model_params['score'] = score  # Store the score separately
                print(f"Model {idx + 1}/{self.n_population} Score: {score}")

                # Update best model parameters if the current model is better
                if score < self.best_score:
                    self.best_score = score
                    self.best_model_params = model_params.copy()  # Store best model parameters

            self.population.sort(key=lambda x: x['score'])  # Sort models by score
            # Exploit best models
            for i in range(self.n_population // 2):
                self.population[i] = self.sample_model()  # Replace with new samples

            # Reintroduce diversity
            for i in range(self.n_population // 2, self.n_population):
                self.population[i] = self.mutate_model(self.population[i])
        
        # Append the best parameters and metrics to CSV after training
        self.append_best_params_to_csv('Hgbrt_PBT', self.best_model_params)

    def mutate_model(self, model_params):
        mutation_rate = 0.1
        for key in model_params.keys():
            if np.random.rand() < mutation_rate:
                if 'learning_rate' in key:
                    model_params[key] = np.random.uniform(0.001, 0.5)
                elif 'max_iter' in key:
                    model_params[key] = np.random.randint(5, 500)
                elif 'max_leaf_nodes' in key:
                    model_params[key] = np.random.randint(5, 100)
                elif 'max_depth' in key:
                    model_params[key] = np.random.randint(5, 30)
                elif 'min_samples_leaf' in key:
                    model_params[key] = np.random.randint(5, 50)
                elif 'l2_regularization' in key:
                    model_params[key] = np.random.uniform(0, 2)
        return model_params

    def append_best_params_to_csv(self, model_name, best_params):
        # Use the same logic to append the best parameters to CSV as before
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
            'min_samples_leaf': [best_params.get('min_samples_leaf', 'None')],
            'Best Score': [self.best_score],  # Include best score
        }

        df_params = pd.DataFrame(ordered_params)
        file_path = '../data/model_best_params.csv'

        if not os.path.isfile(file_path):
            df_params.to_csv(file_path, mode='w', header=True, index=False)
        else:
            df_params.to_csv(file_path, mode='a', header=False, index=False)

# Run Population-Based Training
pbt = PBT(n_generations=10, n_population=10)
pbt.optimize()

# After the optimization is complete, you can also train the model with the best parameters
best_model = HistGradientBoostingRegressor(**{k: v for k, v in pbt.best_model_params.items() if k != 'score'}, random_state=random_seed)
best_model.fit(X_train, y_train)

# Make predictions and calculate metrics for the best model from PBT
y_pred_best = best_model.predict(X_test)
best_metrics = calculate_metrics(y_test, y_pred_best)

# Append best metrics to CSV for PBT
completion_time_best = time.time() - start_time  # Assuming start_time is defined at the start
append_metrics_to_csv('Hgbrt_PBT', best_metrics, completion_time_best)
print("Best Hyperparameters (PBT): ", pbt.best_model_params)
print("Best Metrics (PBT): ", best_metrics)
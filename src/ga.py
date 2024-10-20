import numpy as np
from random import random, uniform
from typing import List, Callable, Tuple
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import os
import time  # Added for time tracking

Genome = List[float]
Population = List[Genome]
FitnessFunc = Callable[[Genome, LGBMRegressor, np.ndarray, np.ndarray], float]

# Generate a genome with real values between the specified bounds
def generate_genome(bounds: np.ndarray) -> Genome:
    return [uniform(bounds[i][0], bounds[i][1]) for i in range(len(bounds))]

# Generate a population of genomes
def generate_population(size: int, bounds: np.ndarray) -> Population:
    return [generate_genome(bounds) for _ in range(size)]

# Single-point crossover for real-valued genomes
def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    p = np.random.randint(1, len(a) - 1)
    return a[:p] + b[p:], b[:p] + a[p:]

# Mutation for real-valued genomes
def mutation(genome: Genome, bounds: np.ndarray, probability: float = 0.1) -> Genome:
    for i in range(len(genome)):
        if random() < probability:
            genome[i] = uniform(bounds[i][0], bounds[i][1])  # Random mutation within bounds
    return genome

# Fitness function that evaluates the model (LightGBM) using NMSE
def fitness_function(genome: Genome, model: LGBMRegressor, X_train: np.ndarray, y_train: np.ndarray) -> float:
    params = {
        'learning_rate': genome[0],
        'n_estimators': int(genome[1]),
        'num_leaves': int(genome[2]),
        'max_depth': int(genome[3]),
        'min_data_in_leaf': int(genome[4]),
        'lambda_l1': genome[5],
        'lambda_l2': genome[6]
    }
    
    model.set_params(**params)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
    return -np.mean(scores)  # Return NMSE (lower is better)

# Sort population based on fitness (lower is better for NMSE)
def sort_population(population: Population, fitness_func: FitnessFunc, model: LGBMRegressor, X_train: np.ndarray, y_train: np.ndarray) -> Population:
    return sorted(population, key=lambda genome: fitness_func(genome, model, X_train, y_train))

# Run the genetic algorithm for real-valued hyperparameter optimization
def run_evolution(
        population_size: int,
        bounds: np.ndarray,
        model: LGBMRegressor,
        X_train: np.ndarray,
        y_train: np.ndarray,
        fitness_func: FitnessFunc,
        fitness_limit: float,
        generation_limit: int = 50,
        mutation_probability: float = 0.1) -> Tuple[Genome, float]:
    
    population = generate_population(population_size, bounds)
    
    start_time = time.time()  # Track the start time
    
    for generation in range(generation_limit):
        print(f"Generation {generation + 1}/{generation_limit}")
        # Sort the population based on fitness
        population = sort_population(population, fitness_func, model, X_train, y_train)
        
        # Check if we reached the fitness limit
        best_genome = population[0]
        best_fitness = fitness_func(best_genome, model, X_train, y_train)
        
        if best_fitness <= fitness_limit:
            break
        
        next_generation = population[:2]  # Preserve the top 2 fittest individuals
        
        # Create the rest of the next generation through crossover and mutation
        for _ in range((population_size // 2) - 1):
            parents = population[np.random.randint(0, population_size)], population[np.random.randint(0, population_size)]
            offspring_a, offspring_b = single_point_crossover(parents[0], parents[1])
            offspring_a = mutation(offspring_a, bounds, mutation_probability)
            offspring_b = mutation(offspring_b, bounds, mutation_probability)
            next_generation += [offspring_a, offspring_b]
        
        population = next_generation
    
    end_time = time.time()  # Track the end time
    completion_time = end_time - start_time  # Calculate completion time
    
    best_genome = population[0]
    best_fitness = fitness_function(best_genome, model, X_train, y_train)
    
    return best_genome, best_fitness, completion_time

# Function to calculate metrics
def calculate_metrics(y_test: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float, float]:
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, rmse, mae, r2, mape

# Function to append metrics to CSV
def append_metrics_to_csv(model_name: str, metrics: Tuple[float, float, float, float, float], completion_time: float, model_category: str = 'Boosting Models'):
    column_order = ['Model Name', 'Model Category', 'MSE', 'RMSE', 'MAE', 'R2', 'MAPE', 'Completion Time']
    metrics_dict = {
        'Model Name': [model_name],
        'Model Category': [model_category],
        'MSE': [metrics[0]],
        'RMSE': [metrics[1]],
        'MAE': [metrics[2]],
        'R2': [metrics[3]],
        'MAPE': [metrics[4]],
        'Completion Time': [completion_time]
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

if __name__ == "__main__":
    # Load the dataset
    data = pd.read_csv("../data/augmented_dataset.csv")
    X = data[['Application_Type', 'Signal_Strength', 'Latency', 'Required_Bandwidth', 'Allocated_Bandwidth']]
    y = data['Resource_Allocation']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the bounds for the hyperparameters
    bounds = np.array([
        [0.05, 0.09],   # learning_rate
        [100, 250],     # n_estimators
        [5, 40],        # num_leaves
        [5, 20],        # max_depth
        [45, 75],       # min_data_in_leaf
        [3, 4],         # lambda_l1
        [3, 4]          # lambda_l2
    ])

    # Initialize the model
    model = LGBMRegressor(random_state=42,verbosity=-1)

    # Run the genetic algorithm
    best_genome, best_fitness, completion_time = run_evolution(
        population_size=20,
        bounds=bounds,
        model=model,
        X_train=X_train.values,
        y_train=y_train.values,
        fitness_func=fitness_function,
        fitness_limit=0.01,  # Desired fitness level (NMSE)
        generation_limit=200
    )

    # Set the best found parameters to the model
    best_params = {
        'learning_rate': best_genome[0],
        'n_estimators': int(best_genome[1]),
        'num_leaves': int(best_genome[2]),
        'max_depth': int(best_genome[3]),
        'min_data_in_leaf': int(best_genome[4]),
        'lambda_l1': best_genome[5],
        'lambda_l2': best_genome[6]
    }
    model.set_params(**best_params)

    # Fit the model with the best parameters
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_train)

    # Calculate metrics
    metrics = calculate_metrics(y_train, y_pred)

    # Append metrics to CSV
    append_metrics_to_csv("LightGBM_GA", metrics, completion_time)

    # Append best parameters to CSV
    append_best_params_to_csv("LightGBM_GA", best_params)

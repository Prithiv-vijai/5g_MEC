import numpy as np
from random import random, uniform
from typing import List, Callable, Tuple
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import os
import time  # Added for time tracking

Genome = List[float]
Population = List[Genome]
FitnessFunc = Callable[[Genome, HistGradientBoostingRegressor, np.ndarray, np.ndarray], float]

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

# Fitness function that evaluates the model (HGBRT) using NMSE
def fitness_function(genome: Genome, model: HistGradientBoostingRegressor, X_train: np.ndarray, y_train: np.ndarray) -> float:
    params = {
        'learning_rate': genome[0],
        'max_iter': int(genome[1]),
        'max_leaf_nodes': int(genome[2]),
        'max_depth': int(genome[3]),
        'min_samples_leaf': int(genome[4]),
        'l2_regularization': genome[5]
    }
    
    model.set_params(**params)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    return -np.mean(scores)  # Return NMSE (lower is better)

# Sort population based on fitness (lower is better for NMSE)
def sort_population(population: Population, fitness_func: FitnessFunc, model: HistGradientBoostingRegressor, X_train: np.ndarray, y_train: np.ndarray) -> Population:
    return sorted(population, key=lambda genome: fitness_func(genome, model, X_train, y_train))

# Run the genetic algorithm for real-valued hyperparameter optimization
def run_evolution(
        population_size: int,
        bounds: np.ndarray,
        model: HistGradientBoostingRegressor,
        X_train: np.ndarray,
        y_train: np.ndarray,
        fitness_func: FitnessFunc,
        fitness_limit: float,
        generation_limit: int = 50,
        mutation_probability: float = 0.1) -> Tuple[Genome, float]:
    
    population = generate_population(population_size, bounds)
    
    start_time = time.time()  # Track the start time
    
    for generation in range(generation_limit):
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
    best_fitness = fitness_func(best_genome, model, X_train, y_train)
    
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
def append_best_params_to_csv(model_name: str, best_params: dict, completion_time: float):
    df_params = pd.DataFrame([best_params])
    df_params.insert(0, 'Model Name', model_name)
    df_params.insert(1, 'Completion Time', completion_time)
    
    file_path = "../data/model_best_params.csv"
    if not os.path.isfile(file_path):
        df_params.to_csv(file_path, mode='w', header=True, index=False)
    else:
        df_params.to_csv(file_path, mode='a', header=False, index=False)

if __name__ == "__main__":
    # Load the dataset
    data = pd.read_csv("../data/augmented_datasett.csv")
    X = data[['Application_Type', 'Signal_Strength', 'Latency', 'Required_Bandwidth', 'Allocated_Bandwidth']]
    y = data['Resource_Allocation']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the bounds for the hyperparameters
    bounds = np.array([
        [0.001, 0.5],    # learning_rate
        [100, 500],      # max_iter
        [20, 100],       # max_leaf_nodes
        [5, 25],         # max_depth
        [10, 50],        # min_samples_leaf
        [0, 2]           # l2_regularization
    ])

    # Initialize the model
    model = HistGradientBoostingRegressor(random_state=42)

    # Run the genetic algorithm
    best_genome, best_fitness, completion_time = run_evolution(
        population_size=100,
        bounds=bounds,
        model=model,
        X_train=X_train.values,
        y_train=y_train.values,
        fitness_func=fitness_function,
        fitness_limit=0.01,  # Desired fitness level (NMSE)
        generation_limit=50
    )

    # Set the best found parameters to the model
    best_params = {
        'learning_rate': best_genome[0],
        'max_iter': int(best_genome[1]),
        'max_leaf_nodes': int(best_genome[2]),
        'max_depth': int(best_genome[3]),
        'min_samples_leaf': int(best_genome[4]),
        'l2_regularization': best_genome[5]
    }

    model.set_params(**best_params)
    model.fit(X_train, y_train)

    # Evaluate on the test set
    y_pred = model.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_pred)
    print("Test Metrics - MSE: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}, R2: {:.4f}, MAPE: {:.4f}"
          .format(*test_metrics))

    # Append the metrics and best parameters to the CSV files
    append_metrics_to_csv(
        model_name="HistGradientBoostingRegressor",
        metrics=test_metrics,
        completion_time=completion_time
    )
    append_best_params_to_csv(
        model_name="HistGradientBoostingRegressor",
        best_params=best_params,
        completion_time=completion_time
    )

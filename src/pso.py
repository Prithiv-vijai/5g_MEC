import numpy as np
import pandas as pd
import time
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import os

# Constants for Particle Swarm Optimization
DIMENSIONS = 6              # Number of dimensions (hyperparameters)
B_LO = [0.001, 100, 20, 5, 10, 0]  # Lower boundary of search space for hyperparameters
B_HI = [0.5, 500, 100, 25, 50, 2]  # Upper boundary of search space for hyperparameters

POPULATION = 20             # Number of particles in the swarm
V_MAX = 0.1                 # Maximum velocity value
PERSONAL_C = 2.0            # Personal coefficient factor
SOCIAL_C = 2.0              # Social coefficient factor
CONVERGENCE = 0.001         # Convergence threshold
MAX_ITER = 50               # Maximum number of iterations
NO_IMPROVEMENT_LIMIT = 5    # Number of iterations with no improvement to stop early

# Load the dataset
data = pd.read_csv("../data/augmented_dataset.csv")

# Define features and target
X = data[['Application_Type', 'Signal_Strength', 'Latency', 'Required_Bandwidth', 'Allocated_Bandwidth']]
y = data['Resource_Allocation']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to calculate metrics
def calculate_metrics(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, rmse, mae, r2, mape

# Function to append metrics to CSV file
def append_metrics_to_csv(model_name, metrics, completion_time=None, model_category='Boosting Models'):
    column_order = ['Model Name', 'Model Category', 'MSE', 'RMSE', 'MAE', 'R2', 'MAPE', 'Completion Time']
    metrics_dict = {
        'Model Name': [model_name],
        'Model Category': [model_category],
        'MSE': [metrics[0]],
        'RMSE': [metrics[1]],
        'MAE': [metrics[2]],
        'R2': [metrics[3]],
        'MAPE': [metrics[4]],
        'Completion Time': [completion_time],
    }
    df_metrics = pd.DataFrame(metrics_dict)
    file_path = "../data/model_performance_metrics.csv"
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Append to CSV or create if it doesn't exist
    df_metrics.to_csv(file_path, mode='a', header=not os.path.isfile(file_path), index=False, columns=column_order)

# Function to save best parameters to CSV
def append_best_params_to_csv(model_name, best_params):
    params_dict = {
        'Model Name': [model_name],
        'l2_regularization': [best_params['l2_regularization']],
        'learning_rate': [best_params['learning_rate']],
        'max_depth': [best_params['max_depth']],
        'max_iter': [best_params['max_iter']],
        'max_leaf_nodes': [best_params['max_leaf_nodes']],
        'min_samples_leaf': [best_params['min_samples_leaf']]
    }
    
    df_params = pd.DataFrame(params_dict)
    file_path = '../data/model_best_params.csv'
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Append to CSV or create if it doesn't exist
    df_params.to_csv(file_path, mode='a', header=not os.path.isfile(file_path), index=False)

# Class representing a particle in the swarm
class Particle:
    def __init__(self, x, z, velocity):
        self.pos = np.clip(x, B_LO, B_HI)   # Ensure initial position is within bounds
        self.pos_z = z
        self.velocity = velocity
        self.best_pos = self.pos.copy()  
        self.best_pos_z = z   # Initialize best_pos_z with the initial cost value

# Class representing the swarm
class Swarm:
    def __init__(self, pop, v_max):
        self.particles = []               
        self.best_pos = None              
        self.best_pos_z = float('inf')    
        
        # Initialize particles
        for _ in range(pop):
            x = np.random.uniform(B_LO, B_HI)  
            z = self.evaluate_cost(x)          
            velocity = np.random.uniform(-v_max, v_max, DIMENSIONS)  
            particle = Particle(x, z, velocity)
            self.particles.append(particle)
            
            if particle.pos_z < self.best_pos_z:
                self.best_pos = particle.pos.copy()
                self.best_pos_z = particle.pos_z

    def evaluate_cost(self, params):
        model = HistGradientBoostingRegressor(
            learning_rate=params[0],
            max_iter=int(params[1]),
            max_leaf_nodes=int(params[2]),
            max_depth=int(params[3]),
            min_samples_leaf=int(params[4]),
            l2_regularization=params[5],
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse

# Particle Swarm Optimization algorithm
def particle_swarm_optimization():
    swarm = Swarm(POPULATION, V_MAX)
    inertia_weight = 0.5

    curr_iter = 0
    start_time = time.time()  
    no_improvement_counter = 0
    previous_best_cost = swarm.best_pos_z

    while curr_iter < MAX_ITER:
        for particle in swarm.particles:
            r1, r2 = np.random.rand(2)
            
            # Update velocity
            personal_coefficient = PERSONAL_C * r1 * (particle.best_pos - particle.pos)
            social_coefficient = SOCIAL_C * r2 * (swarm.best_pos - particle.pos)
            particle.velocity = inertia_weight * particle.velocity + personal_coefficient + social_coefficient
            particle.velocity = np.clip(particle.velocity, -V_MAX, V_MAX)

            # Update position
            particle.pos += particle.velocity
            particle.pos = np.clip(particle.pos, B_LO, B_HI)
            particle.pos_z = swarm.evaluate_cost(particle.pos)

            # Update personal best
            if particle.pos_z < particle.best_pos_z:
                particle.best_pos = particle.pos.copy()
                particle.best_pos_z = particle.pos_z

                # Update global best
                if particle.pos_z < swarm.best_pos_z:
                    swarm.best_pos = particle.pos.copy()
                    swarm.best_pos_z = particle.pos_z

        # Update inertia weight
        inertia_weight *= 0.99

        print(f"Iteration {curr_iter + 1}/{MAX_ITER}: Best Cost = {swarm.best_pos_z}")

        if abs(previous_best_cost - swarm.best_pos_z) < CONVERGENCE:
            no_improvement_counter += 1
            if no_improvement_counter >= NO_IMPROVEMENT_LIMIT:
                print(f"No significant improvement for {NO_IMPROVEMENT_LIMIT} consecutive iterations. Stopping early.")
                break
        else:
            no_improvement_counter = 0
            previous_best_cost = swarm.best_pos_z

        curr_iter += 1

    end_time = time.time()  
    completion_time = end_time - start_time

    print("Best Parameters found: ", swarm.best_pos)
    print("Best Cost: ", swarm.best_pos_z)

    best_params = {
        'learning_rate': swarm.best_pos[0],
        'max_iter': int(swarm.best_pos[1]),
        'max_leaf_nodes': int(swarm.best_pos[2]),
        'max_depth': int(swarm.best_pos[3]),
        'min_samples_leaf': int(swarm.best_pos[4]),
        'l2_regularization': swarm.best_pos[5]
    }

    model = HistGradientBoostingRegressor(
        learning_rate=best_params['learning_rate'],
        max_iter=best_params['max_iter'],
        max_leaf_nodes=best_params['max_leaf_nodes'],
        max_depth=best_params['max_depth'],
        min_samples_leaf=best_params['min_samples_leaf'],
        l2_regularization=best_params['l2_regularization'],
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = calculate_metrics(y_test, y_pred)
    append_metrics_to_csv('Hgbrt_PSO', metrics, completion_time)
    append_best_params_to_csv('Hgbrt_PSO', best_params)

    print(f"MSE: {metrics[0]}")
    print(f"RMSE: {metrics[1]}")
    print(f"MAE: {metrics[2]}")
    print(f"R2: {metrics[3]}")
    print(f"MAPE: {metrics[4]}")

if __name__ == "__main__":
    particle_swarm_optimization()

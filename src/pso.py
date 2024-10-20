import numpy as np
import pandas as pd
import time
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import os

# Constants for Particle Swarm Optimization
DIMENSIONS = 7              # Number of dimensions (hyperparameters for LightGBM)
B_LO = [100, 0.05, 5, 5, 35, 1, 1]  # Lower boundary of search space for hyperparameters
B_HI = [250, 0.09, 40, 20, 75, 3, 3]  # Upper boundary of search space for hyperparameters

POPULATION = 20             # Number of particles in the swarm
V_MAX = 0.1                 # Maximum velocity value
PERSONAL_C = 2.0            # Personal coefficient factor
SOCIAL_C = 2.0              # Social coefficient factor
CONVERGENCE = 0.001         # Convergence threshold
MAX_ITER = 200              # Maximum number of iterations

RANDOM_SEED = 42            # Seed for reproducibility

# Set random seed for NumPy
np.random.seed(RANDOM_SEED)

# Load the dataset
data = pd.read_csv("../data/augmented_dataset.csv")

# Define features and target
X = data[['Application_Type', 'Signal_Strength', 'Latency', 'Required_Bandwidth', 'Allocated_Bandwidth']]
y = data['Resource_Allocation']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)

# Function to calculate metrics
def calculate_metrics(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, rmse, mae, r2, mape

# Function to append metrics to CSV file
def append_metrics_to_csv(model_name, metrics, completion_time=None, model_category='Optimization Models'):
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
        model = LGBMRegressor(
            n_estimators=int(params[0]),
            learning_rate=params[1],
            num_leaves=int(params[2]),
            max_depth=int(params[3]),
            min_data_in_leaf=int(params[4]),
            lambda_l1=params[5],
            lambda_l2=params[6],
            random_state=RANDOM_SEED,
            verbose=-1
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

        curr_iter += 1

    end_time = time.time()  
    completion_time = end_time - start_time

    print("Best Parameters found: ", swarm.best_pos)
    print("Best Cost: ", swarm.best_pos_z)

    best_params = {
        'n_estimators': int(swarm.best_pos[0]),
        'learning_rate': swarm.best_pos[1],
        'num_leaves': int(swarm.best_pos[2]),
        'max_depth': int(swarm.best_pos[3]),
        'min_data_in_leaf': int(swarm.best_pos[4]),
        'lambda_l1': swarm.best_pos[5],
        'lambda_l2': swarm.best_pos[6]
    }

    model = LGBMRegressor(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        num_leaves=best_params['num_leaves'],
        max_depth=best_params['max_depth'],
        min_data_in_leaf=best_params['min_data_in_leaf'],
        lambda_l1=best_params['lambda_l1'],
        lambda_l2=best_params['lambda_l2'],
        random_state=RANDOM_SEED,
        verbosity=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)

    metrics = calculate_metrics(y_train, y_pred)
    append_metrics_to_csv('LightGBM_PSO', metrics, completion_time)
    append_best_params_to_csv('LightGBM_PSO', best_params)

    print(f"MSE: {metrics[0]}")
    print(f"RMSE: {metrics[1]}")
    print(f"MAE: {metrics[2]}")
    print(f"R2: {metrics[3]}")
    print(f"MAPE: {metrics[4]}")

if __name__ == "__main__":
    particle_swarm_optimization()

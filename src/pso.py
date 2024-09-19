import numpy as np
import pandas as pd
import math
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import time

DIMENSIONS = 6              # Number of dimensions for hyperparameters
GLOBAL_BEST = 0             # Global Best of Cost function
B_LO = [0.001, 100, 20, 5, 10, 0]  # Lower boundary of search space
B_HI = [0.5, 500, 100, 25, 50, 2]  # Upper boundary of search space

POPULATION = 20             # Number of particles in the swarm
V_MAX = 0.1                 # Maximum velocity value
PERSONAL_C = 2.0            # Personal coefficient factor
SOCIAL_C = 2.0              # Social coefficient factor
CONVERGENCE = 0.001         # Convergence value
MAX_ITER = 50              # Maximum number of iterations

# Load the dataset
data = pd.read_csv("../data/augmented_dataset.csv")

# Define features and target
X = data[['Application_Type', 'Signal_Strength', 'Latency', 'Required_Bandwidth', 'Allocated_Bandwidth']]
y = data['Resource_Allocation']

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

# Particle class
class Particle():
    def __init__(self, x, z, velocity):
        self.pos = x
        self.pos_z = z
        self.velocity = velocity
        self.best_pos = self.pos.copy()

class Swarm():
    def __init__(self, pop, v_max):
        self.particles = []             # List of particles in the swarm
        self.best_pos = None            # Best particle of the swarm
        self.best_pos_z = float('inf')  # Best particle of the swarm

        for _ in range(pop):
            x = np.random.uniform(B_LO, B_HI)
            z = self.evaluate_cost(x)
            velocity = np.random.rand(DIMENSIONS) * v_max
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

def particle_swarm_optimization():
    # Initialize plotting variables
    x = np.linspace(B_LO[0], B_HI[0], 50)
    y = np.linspace(B_LO[1], B_HI[1], 50)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure("Particle Swarm Optimization")

    # Initialize swarm
    swarm = Swarm(POPULATION, V_MAX)

    # Initialize inertia weight
    inertia_weight = 0.5 + (np.random.rand()/2)
    
    curr_iter = 0
    start_time = time.time()  # Start time
    
    while curr_iter < MAX_ITER:
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)
        Z = np.array([[swarm.evaluate_cost([xi, yi, 20, 5, 10, 0]) for xi in x] for yi in y])
        c = ax.contourf(X, Y, Z, cmap='viridis')
        plt.colorbar(c)

        for particle in swarm.particles:
            for i in range(DIMENSIONS):
                r1 = np.random.uniform(0, 1)
                r2 = np.random.uniform(0, 1)
                
                # Update particle's velocity
                personal_coefficient = PERSONAL_C * r1 * (particle.best_pos[i] - particle.pos[i])
                social_coefficient = SOCIAL_C * r2 * (swarm.best_pos[i] - particle.pos[i])
                new_velocity = inertia_weight * particle.velocity[i] + personal_coefficient + social_coefficient

                # Check if velocity is exceeded
                if new_velocity > V_MAX:
                    new_velocity = V_MAX
                elif new_velocity < -V_MAX:
                    new_velocity = -V_MAX

                particle.velocity[i] = new_velocity

            ax.scatter(particle.pos[0], particle.pos[1], marker='*', c='r')
            ax.arrow(particle.pos[0], particle.pos[1], particle.velocity[0], particle.velocity[1], head_width=0.1, head_length=0.1, color='k')

            # Update particle's current position
            particle.pos += particle.velocity
            particle.pos_z = swarm.evaluate_cost(particle.pos)

            # Update particle's best known position
            if particle.pos_z < swarm.evaluate_cost(particle.best_pos):
                particle.best_pos = particle.pos.copy()

                # Update swarm's best known position
                if particle.pos_z < swarm.best_pos_z:
                    swarm.best_pos = particle.pos.copy()
                    swarm.best_pos_z = particle.pos_z
                    
            # Check if particle is within boundaries
            for dim in range(DIMENSIONS):
                if particle.pos[dim] > B_HI[dim]:
                    particle.pos[dim] = np.random.uniform(B_LO[dim], B_HI[dim])
                    particle.pos_z = swarm.evaluate_cost(particle.pos)
                if particle.pos[dim] < B_LO[dim]:
                    particle.pos[dim] = np.random.uniform(B_LO[dim], B_HI[dim])
                    particle.pos_z = swarm.evaluate_cost(particle.pos)

        plt.subplots_adjust(right=0.95)
        plt.pause(0.00001)

        # Check for convergence
        if abs(swarm.best_pos_z - GLOBAL_BEST) < CONVERGENCE:
            print("The swarm has met convergence criteria after " + str(curr_iter) + " iterations.")
            break
        curr_iter += 1

    end_time = time.time()  # End time
    completion_time = end_time - start_time

    # Display final results
    print("Best Parameters found: ", swarm.best_pos)
    print("Best Cost: ", swarm.best_pos_z)
    append_metrics_to_csv('Particle Swarm Optimization', calculate_metrics(y_test, HistGradientBoostingRegressor(
        learning_rate=swarm.best_pos[0],
        max_iter=int(swarm.best_pos[1]),
        max_leaf_nodes=int(swarm.best_pos[2]),
        max_depth=int(swarm.best_pos[3]),
        min_samples_leaf=int(swarm.best_pos[4]),
        l2_regularization=swarm.best_pos[5],
        random_state=42
    ).predict(X_test)), completion_time)

    plt.show()

particle_swarm_optimization()
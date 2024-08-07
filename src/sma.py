import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset from a CSV file
data = pd.read_csv('../data/augmented_dataset.csv')

# Define features and target
X = data[['Application_Type', 'Signal_Strength', 'Latency', 'Required_Bandwidth', 'Allocated_Bandwidth']]
y = data['Resource_Allocation']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def fitness_function(params):
    learning_rate, max_iter, max_leaf_nodes, max_depth, min_samples_leaf, l2_regularization = params
    model = HistGradientBoostingRegressor(
        learning_rate=learning_rate,
        max_iter=int(max_iter),
        max_leaf_nodes=int(max_leaf_nodes),
        max_depth=int(max_depth) if not np.isnan(max_depth) else None,
        min_samples_leaf=int(min_samples_leaf),
        l2_regularization=l2_regularization
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

def slime_mould_algorithm(fitness_function, dim, bounds, n_agents=30, max_iter=100):
    # Initialize the population
    population = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(n_agents, dim))
    fitness = np.array([fitness_function(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx]
    best_fitness = fitness[best_idx]

    for t in range(max_iter):
        mean_position = np.mean(population, axis=0)
        for i in range(n_agents):
            r = np.random.rand()
            if r < 0.5:
                population[i] = mean_position + np.random.normal(0, 0.1, dim)
            else:
                population[i] = best_solution + np.random.normal(0, 0.1, dim)
        
        population = np.clip(population, [b[0] for b in bounds], [b[1] for b in bounds])
        fitness = np.array([fitness_function(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
    
    return best_solution, best_fitness

# Define the parameter bounds
bounds = [
    (0.01, 0.2),        # learning_rate
    (100, 300),         # max_iter
    (31, 70),           # max_leaf_nodes
    (10, 20),           # max_depth
    (20, 40),           # min_samples_leaf
    (0, 1)              # l2_regularization
]

# Run the Slime Mould Algorithm
best_solution, best_fitness = slime_mould_algorithm(fitness_function, dim=6, bounds=bounds, n_agents=30, max_iter=100)

# Train the best model found by SMA
best_model = HistGradientBoostingRegressor(
    learning_rate=best_solution[0],
    max_iter=int(best_solution[1]),
    max_leaf_nodes=int(best_solution[2]),
    max_depth=int(best_solution[3]) if not np.isnan(best_solution[3]) else None,
    min_samples_leaf=int(best_solution[4]),
    l2_regularization=best_solution[5]
)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = best_model.score(X_test, y_test)
adjusted_r2 = 1 - (1-r2) * (len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
mape = mean_absolute_percentage_error(y_test, y_pred)

# Create an image with the metrics
plt.figure(figsize=(10, 6))
plt.text(0.5, 0.9, f"Best Parameters from SMA:", ha='center', va='center', fontsize=14, weight='bold')
plt.text(0.5, 0.8, f"Learning Rate: {best_solution[0]:.4f}", ha='center', va='center', fontsize=12)
plt.text(0.5, 0.7, f"Max Iterations: {int(best_solution[1])}", ha='center', va='center', fontsize=12)
plt.text(0.5, 0.6, f"Max Leaf Nodes: {int(best_solution[2])}", ha='center', va='center', fontsize=12)
plt.text(0.5, 0.5, f"Max Depth: {int(best_solution[3]) if not np.isnan(best_solution[3]) else 'None'}", ha='center', va='center', fontsize=12)
plt.text(0.5, 0.4, f"Min Samples Leaf: {int(best_solution[4])}", ha='center', va='center', fontsize=12)
plt.text(0.5, 0.3, f"L2 Regularization: {best_solution[5]:.4f}", ha='center', va='center', fontsize=12)

plt.text(0.5, 0.1, f"MSE: {mse:.4f}", ha='center', va='center', fontsize=12)
plt.text(0.5, 0.0, f"RMSE: {rmse:.4f}", ha='center', va='center', fontsize=12)
plt.text(0.5, -0.1, f"R2: {r2:.4f}", ha='center', va='center', fontsize=12)
plt.text(0.5, -0.2, f"Adjusted R2: {adjusted_r2:.4f}", ha='center', va='center', fontsize=12)
plt.text(0.5, -0.3, f"MAPE: {mape:.4f}", ha='center', va='center', fontsize=12)

plt.axis('off')
plt.tight_layout()
plt.savefig('../graphs/model_output/extras/sma_model_metrics_image.png')
plt.show()

print("Metrics image saved to '../results/model_metrics_image.png'.")

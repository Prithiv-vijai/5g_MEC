import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.optimize import differential_evolution
import os
import time
from tpot import TPOTRegressor
from skopt import gp_minimize

# Load the dataset from a CSV file
data = pd.read_csv('../data/augmented_dataset.csv')

iter=50


# Define features and target
X = data[['Application_Type', 'Signal_Strength', 'Latency', 'Required_Bandwidth', 'Allocated_Bandwidth']]
y = data['Resource_Allocation']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = HistGradientBoostingRegressor(random_state=42)

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

# Function to evaluate the model using K-Fold Cross-Validation
def evaluate_model(model, X_train, y_train, X_test, y_test):
    # K-Fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return calculate_metrics(y_test, y_pred), np.mean(np.sqrt(-cv_scores))

# AutoML using TPOT with enhanced configuration
def automl_tpot():
    print("Starting TPOT AutoML with 5-fold CV...")
    start_time = time.time()
    model = TPOTRegressor(generations=25, population_size=30, random_state=40, verbosity=2, cv=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics, avg_cv_rmse = evaluate_model(model, X_train, y_train, X_test, y_test)
    print(f"TPOT Avg CV RMSE: {avg_cv_rmse}")
    append_metrics_to_csv('TPOT', metrics, time.time() - start_time, model_category='AutoML')
    print("TPOT Optimization completed.\n")

# Gradient-Based Optimization with Skopt
def gradient_based_optimization():
    print("Starting Gradient-Based Optimization with Skopt (5-fold CV)...")

    def objective_function(params):
        learning_rate, max_iter, max_leaf_nodes, max_depth, min_samples_leaf, l2_regularization = params
        model.set_params(learning_rate=learning_rate,
                         max_iter=int(max_iter),
                         max_leaf_nodes=int(max_leaf_nodes),
                         max_depth=int(max_depth),
                         min_samples_leaf=int(min_samples_leaf),
                         l2_regularization=l2_regularization)
        
        # Apply 5-fold CV
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        mse = -cv_scores.mean()  # Negative MSE to positive
        print(f"Iteration params: {params}, MSE: {mse}")
        return mse

    bounds = [
        (0.001, 0.1),   # learning_rate
        (100, 300),     # max_iter
        (5, 50),        # max_leaf_nodes
        (5, 25),        # max_depth
        (10, 50),       # min_samples_leaf
        (1, 5)          # l2_regularization
    ]

    start_time = time.time()
    res = gp_minimize(objective_function, bounds, n_calls=100, verbose=True)
    best_params = res.x
    model.set_params(learning_rate=best_params[0],
                     max_iter=int(best_params[1]),
                     max_leaf_nodes=int(best_params[2]),
                     max_depth=int(best_params[3]),
                     min_samples_leaf=int(best_params[4]),
                     l2_regularization=best_params[5])
    
    metrics, _ = evaluate_model(model, X_train, y_train, X_test, y_test)
    append_metrics_to_csv('Gradient-Based Optimization', metrics, time.time() - start_time, model_category='Gradient-Based')
    print("Gradient-Based Optimization completed.\n")

# Differential Evolution Optimization
def differential_evolution_optimization():
    print("Starting Differential Evolution Optimization with 5-fold CV...")

    def objective_function(params):
        learning_rate, max_iter, max_leaf_nodes, max_depth, min_samples_leaf, l2_regularization = params
        model.set_params(learning_rate=learning_rate,
                         max_iter=int(max_iter),
                         max_leaf_nodes=int(max_leaf_nodes),
                         max_depth=int(max_depth),
                         min_samples_leaf=int(min_samples_leaf),
                         l2_regularization=l2_regularization)
        
        # Apply 5-fold CV
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        mse = -cv_scores.mean()  # Negative MSE to positive
        print(f"Iteration params: {params}, MSE: {mse}")
        return mse

    bounds = [
        (0.001, 0.1),   # learning_rate
        (100, 300),     # max_iter
        (5, 50),        # max_leaf_nodes
        (5, 25),        # max_depth
        (10, 50),       # min_samples_leaf
        (1, 5)          # l2_regularization
    ]

    start_time = time.time()
    result = differential_evolution(objective_function, bounds, maxiter=100, disp=True)
    best_params = result.x
    model.set_params(learning_rate=best_params[0],
                     max_iter=int(best_params[1]),
                     max_leaf_nodes=int(best_params[2]),
                     max_depth=int(best_params[3]),
                     min_samples_leaf=int(best_params[4]),
                     l2_regularization=best_params[5])

    metrics, _ = evaluate_model(model, X_train, y_train, X_test, y_test)
    append_metrics_to_csv('Differential Evolution', metrics, time.time() - start_time, model_category='Evolutionary')
    print("Differential Evolution Optimization completed.\n")

# Call the optimization functions
# automl_tpot()
differential_evolution_optimization()
gradient_based_optimization()

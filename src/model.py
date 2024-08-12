import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# Additional libraries for the new models
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Create the directory if it doesn't exist
output_dir = '../graphs/model_output/'
os.makedirs(output_dir, exist_ok=True)

# Load the dataset from a CSV file
df = pd.read_csv('../data/augmented_datasett.csv')

# Define features (X) and target (y)
X = df[['Application_Type', 'Signal_Strength', 'Latency', 'Required_Bandwidth', 'Allocated_Bandwidth']]
y = df['Resource_Allocation']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models grouped by type
model_groups = {
    'Regression Models': {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(max_iter=10000),
        'Lasso Regression': Lasso(max_iter=10000),
        'Elastic Net': ElasticNet(max_iter=10000),
        'Bayesian Ridge': BayesianRidge(),
        'Polynomial Regression': make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
        'SVR': SVR(),
        'MLP Regressor': MLPRegressor(max_iter=10000, random_state=42),
    },
    'Tree-Based Models': {
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42),
    },
    'Boosting Models': {
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'AdaBoost': AdaBoostRegressor(random_state=42),
        'XGBoost': xgb.XGBRegressor(random_state=42),
        'LightGBM': lgb.LGBMRegressor(random_state=42),
        'CatBoost': CatBoostRegressor(verbose=0)
    }
}

# Dictionary to store the results
results = {}
predictions = {}

# Hyperparameter grids for Grid Search
param_grids = {
    'Regression Models': {
        'Linear Regression': {},
        'Ridge Regression': {'alpha': [0.1, 1, 10]},
        'Lasso Regression': {'alpha': [0.01, 0.1, 1, 10]},
        'Elastic Net': {'alpha': [0.1, 1, 10], 'l1_ratio': [0.1, 0.5, 0.9]},
        'Bayesian Ridge': {'alpha_1': [1e-6, 1e-3, 1e-1], 'alpha_2': [1e-6, 1e-3, 1e-1], 'lambda_1': [1e-6, 1e-3, 1e-1], 'lambda_2': [1e-6, 1e-3, 1e-1]},
        'Polynomial Regression': {},
        'SVR': {'C': [0.1, 1, 10], 'epsilon': [0.01, 0.1, 0.5], 'kernel': ['linear', 'poly', 'rbf'], 'gamma': ['scale', 'auto', 0.01, 0.1]},
        'MLP Regressor': {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'activation': ['relu', 'tanh', 'logistic'], 'solver': ['adam', 'sgd', 'lbfgs'], 'alpha': [0.0001, 0.001]}
    },
    'Tree-Based Models': {
        'Decision Tree': {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 5]},
        'Random Forest': {'n_estimators': [100, 200, 500], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 5], 'max_features': ['sqrt', 'log2', None]},
    },
    'Boosting Models': {
        'Gradient Boosting': {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]},
        'AdaBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]},
        'XGBoost': {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7], 'subsample': [0.8, 0.9, 1.0], 'colsample_bytree': [0.8, 0.9, 1.0]},
        'LightGBM': {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7], 'num_leaves': [31, 63, 127]},
        'CatBoost': {'iterations': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2], 'depth': [6, 8, 10]}
    }
}

# Perform Grid Search for each model
for group_name, models in model_groups.items():
    for name, model in models.items():
        if name in param_grids[group_name]:
            grid_search = GridSearchCV(model, param_grids[group_name][name], cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
        else:
            best_model = model
            best_model.fit(X_train, y_train)
        
        y_pred = best_model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        adj_r2 = 1 - (1-r2) * (len(y_test)-1) / (len(y_test) - X_test.shape[1] - 1)

        results[(group_name, name)] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape, 'Adjusted R2': adj_r2}
        predictions[(group_name, name)] = y_pred

# Define metrics to plot
metrics_to_plot = ['MSE', 'RMSE', 'MAE', 'R2', 'MAPE', 'Adjusted R2']

# Function to add rank labels
def add_rank_labels(ax, data, metric):
    sorted_data = data.sort_values(by=metric, ascending=metric not in ['R2', 'Adjusted R2'])
    for rank, (index, row) in enumerate(sorted_data.iterrows(), 1):
        bar = ax.patches[index]
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'#{rank}', 
                color='green' if rank == 1 else 'orange' if rank == 2 else 'red', 
                ha='center', va='bottom', fontsize=12, fontweight='bold')

# Generate separate plots for each group of models
for group_name, models in model_groups.items():
    group_results = {name: results[(group_name, name)] for name in models.keys()}
    group_metrics_df = pd.DataFrame(group_results).T
    group_metrics_df.reset_index(inplace=True)
    group_metrics_df = group_metrics_df.rename(columns={'index': 'Model'})

    num_metrics = len(metrics_to_plot)
    num_cols = 3
    num_rows = (num_metrics + num_cols - 1) // num_cols  # Calculate number of rows required

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5), tight_layout=True)
    if num_rows == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics_to_plot):
        row = idx // num_cols
        col = idx % num_cols
        ax = axes[row][col]

        # Sorting models by the current metric
        sorted_group_df = group_metrics_df.sort_values(by=metric, ascending=metric not in ['R2', 'Adjusted R2'])
        sns.barplot(data=sorted_group_df, x='Model', y=metric, ax=ax, palette='viridis')
        ax.set_title(f'{group_name} - {metric}')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        # Add rank labels
        add_rank_labels(ax, sorted_group_df, metric)

    # Remove any empty subplots
    for i in range(num_metrics, num_rows * num_cols):
        fig.delaxes(axes[i // num_cols][i % num_cols])

    plt.savefig(f'{output_dir}{group_name}_metrics_comparison.png')
    plt.close()

# Print results
for (group_name, name), metrics in results.items():
    print(f"{group_name} - {name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    print()

# Save the results to a CSV file
results_df = pd.DataFrame(results).T
results_df.to_csv('../graphs/model_output/optimized_results.csv', index=True)

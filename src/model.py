import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import time  # Import time module for tracking completion time
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb
from sklearn.model_selection import KFold

# Create the directory if it doesn't exist
output_dir = '../graphs/model_output/'
os.makedirs(output_dir, exist_ok=True)

# Load the dataset from a CSV file
df = pd.read_csv('../data/augmented_dataset.csv')

# Define features (X) and target (y)
X = df[['Application_Type', 'Signal_Strength', 'Latency', 'Required_Bandwidth', 'Allocated_Bandwidth']]
y = df['Resource_Allocation']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the models grouped by type
model_groups = {
    'Regression Models': {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Elastic Net': ElasticNet(),
        'Bayesian Ridge': BayesianRidge(),
        'Polynomial Regression': make_pipeline(PolynomialFeatures(degree=3), LinearRegression()),
        'SVR': SVR(),
        'MLP Regressor': MLPRegressor(random_state=42),
    },
    'Tree-Based Models': {
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42),
    },
    'Boosting Models': {
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'AdaBoost': AdaBoostRegressor(random_state=42),
        'XGBoost': xgb.XGBRegressor(random_state=42),
        'Hgbrt': HistGradientBoostingRegressor(random_state=42),
        'LightGBM': lgb.LGBMRegressor(random_state=42)
    }
}

# Dictionary to store the results
results = {}
predictions = {}

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Train and evaluate each model using cross-validation
for group_name, models in model_groups.items():
    for name, model in models.items():
        start_time = time.time()  # Start time for the model training
        
        # Use cross_val_predict to get cross-validated predictions
        y_pred = cross_val_predict(model, X_train, y_train, cv=kf)

        completion_time = time.time() - start_time  # Calculate completion time

        # Calculate metrics
        mse = mean_squared_error(y_train, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_train, y_pred)
        r2 = r2_score(y_train, y_pred)
        mape = mean_absolute_percentage_error(y_train, y_pred)

        results[(group_name, name)] = {
            'Model Name': name,
            'Model Category': group_name,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'Completion_Time': completion_time  # Add completion time
        }
        predictions[(group_name, name)] = y_pred

# Convert results to a DataFrame
results_df = pd.DataFrame(results).T.reset_index(drop=True)

# Save the results DataFrame to a CSV file
results_df.to_csv('../data/model_performance_metrics.csv', index=False)

# Define metrics to plot, limited to MSE, MAE, and R2
metrics_to_plot = ['MSE', 'MAE', 'R2']  # Restrict to only the specified metrics

# Function to add rank labels
def add_rank_labels(ax, data, metric):
    sorted_data = data.sort_values(by=metric, ascending=metric not in ['R2'])
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
    num_rows = (num_metrics + num_cols - 1) // num_cols  # Ensure enough rows for all metrics

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(24, num_rows * 5))
    axes = axes.ravel()  # Flatten the 2D array of axes for easy iteration

    for i, metric in enumerate(metrics_to_plot):
        sns.barplot(x='Model', y=metric, data=group_metrics_df, ax=axes[i])
        axes[i].set_title(f'{metric} Comparison ({group_name})')
        axes[i].tick_params(axis='x', rotation=45)
        if metric == 'R2':
            axes[i].set_ylim(0, 1.5)
        else:
            axes[i].set_ylim(0, group_metrics_df[metric].max() * 1.1)
        add_rank_labels(axes[i], group_metrics_df, metric)

    # Hide any unused subplots
    for j in range(num_metrics, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{group_name.lower().replace(" ", "_")}_metrics_comparison(augmented).png'))

# Find and print the best values for each metric
best_models = {metric: min(results.items(), key=lambda x: x[1][metric]) for metric in ['MSE', 'RMSE', 'MAE', 'MAPE']}
best_models['R2'] = max(results.items(), key=lambda x: x[1]['R2'])

print("\nBest Models for Each Metric:")
for metric, ((group_name, model_name), metrics) in best_models.items():
    print(f"\nBest Model for {metric}: {model_name} ({group_name})")
    print(f"  {metric}: {metrics[metric]:.4f}")

# Create a DataFrame for top-ranked models from each category
top_models = {}
for group_name, models in model_groups.items():
    if group_name == 'Boosting Models':
        # Special handling for Boosting Models to select the model with the best R2 score
        best_model = max(models.keys(), key=lambda name: results[(group_name, name)]['R2'])
    else:
        # Default handling for other categories
        best_model = min(models.keys(), key=lambda name: results[(group_name, name)]['MAE'])
    
    top_models[group_name] = best_model

# Generate a DataFrame for top-ranked models
top_models_df = pd.DataFrame.from_dict(top_models, orient='index', columns=['Best_Model']).reset_index()
top_models_df.rename(columns={'index': 'Model_Category'}, inplace=True)

print("\nTop Ranked Models:")
print(top_models_df)
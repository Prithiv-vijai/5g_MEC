import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# Create the directory if it doesn't exist
output_dir = '../graphs/model_output'
os.makedirs(output_dir, exist_ok=True)

# Load the dataset from a CSV file
df = pd.read_csv('../data/preprocessed_dataset.csv')

# Define features (X) and target (y)
X = df[['Application_Type', 'Signal_Strength', 'Latency', 'Required_Bandwidth', 'Allocated_Bandwidth']]
y = df['Resource_Allocation']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(max_iter=10000),
    'Lasso Regression': Lasso(max_iter=10000),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'KNN': KNeighborsRegressor(),
    'SVM': SVR(),
    'Polynomial Regression': make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
}

# Dictionary to store the results
results = {}
predictions = {}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    adj_r2 = 1 - (1-r2) * (len(y_test)-1) / (len(y_test) - X_test.shape[1] - 1)
    
    results[name] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape, 'Adjusted R2': adj_r2}
    predictions[name] = y_pred

# Print the comparison of the models
print("Model Comparison:")
for name, metrics in results.items():
    print(f"\n{name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

# Visualize the metrics comparison
metrics_df = pd.DataFrame(results).T
metrics_df.reset_index(inplace=True)
metrics_df = metrics_df.rename(columns={'index': 'Model'})

# Define metrics to plot and calculate the number of rows and columns
metrics_to_plot = ['MSE', 'RMSE', 'MAE', 'R2', 'MAPE', 'Adjusted R2']
num_metrics = len(metrics_to_plot)
num_cols = 3
num_rows = (num_metrics + num_cols - 1) // num_cols  # Ensure enough rows for all metrics

fig, axes = plt.subplots(num_rows, num_cols, figsize=(24, num_rows * 5))
axes = axes.ravel()  # Flatten the 2D array of axes for easy iteration

# Function to add rank labels
def add_rank_labels(ax, data, metric):
    sorted_data = data.sort_values(by=metric, ascending=metric not in ['R2', 'Adjusted R2'])
    for rank, (index, row) in enumerate(sorted_data.iterrows(), 1):
        bar = ax.patches[index]
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'#{rank}', 
                color='green' if rank == 1 else 'orange' if rank == 2 else 'red', 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
        


for i, metric in enumerate(metrics_to_plot):
    sns.barplot(x='Model', y=metric, data=metrics_df, ax=axes[i])
    axes[i].set_title(f'{metric} Comparison')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].set_ylim(0, metrics_df[metric].max() * 1.1) if metric != 'R2' and metric != 'Adjusted R2' else axes[i].set_ylim(metrics_df[metric].min() * 1.1, 1)
    add_rank_labels(axes[i], metrics_df, metric)

# Hide any unused subplots
for j in range(num_metrics, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'))

# Visualize the predicted vs actual values
num_models = len(models)
num_cols = min(num_models, 3)  # Limit to a maximum of 3 columns for better layout
num_rows = (num_models + num_cols - 1) // num_cols  # Calculate rows needed

fig, axes = plt.subplots(num_rows, num_cols, figsize=(24, num_rows * 5))
axes = axes.ravel()

for i, (name, y_pred) in enumerate(predictions.items()):
    ax = axes[i]
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_title(f'{name} Predicted vs Actual')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')

# Hide any unused subplots
for j in range(num_models, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'predicted_vs_actual.png'))

# Find and print the best values for each metric
best_models = {metric: min(results.items(), key=lambda x: x[1][metric]) for metric in ['MSE', 'RMSE', 'MAE', 'MAPE']}
best_models['R2'] = max(results.items(), key=lambda x: x[1]['R2'])
best_models['Adjusted R2'] = max(results.items(), key=lambda x: x[1]['Adjusted R2'])

print("\nBest Models for Each Metric:")
for metric, (name, metrics) in best_models.items():
    print(f"\nBest Model for {metric}: {name}")
    print(f"  {metric}: {metrics[metric]:.4f}")
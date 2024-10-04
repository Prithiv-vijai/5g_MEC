import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb

# Load the dataset from a CSV file
df = pd.read_csv('../data/augmented_dataset.csv')

# Define features (X) and target (y)
X = df[['Application_Type', 'Signal_Strength', 'Latency', 'Required_Bandwidth', 'Allocated_Bandwidth']]
y = df['Resource_Allocation']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models
models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': xgb.XGBRegressor(random_state=42),
    'HGBRT': HistGradientBoostingRegressor(random_state=42)
}

# Dictionary to store the results
results = {}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    results[name] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape
    }

# Convert the results dictionary to a DataFrame for ranking and display
metrics_df = pd.DataFrame(results).T

# Print the metrics for each model
print("\nMetrics for each model:")
print(metrics_df)

# Identify the best model for each metric
print("\nBest models based on each metric:")

for metric in ['MSE', 'RMSE', 'MAE', 'R²', 'MAPE']:
    best_model = metrics_df[metric].idxmin() if metric != 'R²' else metrics_df[metric].idxmax()
    best_value = metrics_df[metric][best_model]
    print(f"Best {metric}: {best_model} with value {best_value:.4f}")

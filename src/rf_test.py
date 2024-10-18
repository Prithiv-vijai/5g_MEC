import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# Load the dataset from a CSV file
data = pd.read_csv('../data/augmented_dataset.csv')

# Define features and target
X = data[['Application_Type', 'Signal_Strength', 'Latency', 'Required_Bandwidth', 'Allocated_Bandwidth']]
y = data['Resource_Allocation']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mse, rmse, mae, r2, mape

# Manually set hyperparameters for the RandomForestRegressor
manual_params = {
    'n_estimators': 490,
    'max_depth': 15,
    'min_samples_leaf': 5,
    'max_leaf_nodes': 50,
    'random_state': 42
}

# Initialize the model with the manually set hyperparameters
model = RandomForestRegressor(
    n_estimators=manual_params['n_estimators'],
    max_depth=manual_params['max_depth'],
    min_samples_leaf=manual_params['min_samples_leaf'],
    max_leaf_nodes=manual_params['max_leaf_nodes'],
    random_state=manual_params['random_state']
)

# Train the model
model.fit(X_train, y_train)

# Make predictions on both training and testing sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate metrics for training and testing sets
train_metrics = calculate_metrics(y_train, y_train_pred)
test_metrics = calculate_metrics(y_test, y_test_pred)

# Print the metrics
print("Training Metrics:")
print(f"MSE: {train_metrics[0]:.4f}, RMSE: {train_metrics[1]:.4f}, MAE: {train_metrics[2]:.4f}, R2: {train_metrics[3]:.4f}, MAPE: {train_metrics[4]:.4f}")

print("\nTesting Metrics:")
print(f"MSE: {test_metrics[0]:.4f}, RMSE: {test_metrics[1]:.4f}, MAE: {test_metrics[2]:.4f}, R2: {test_metrics[3]:.4f}, MAPE: {test_metrics[4]:.4f}")

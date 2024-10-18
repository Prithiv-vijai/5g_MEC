import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
import matplotlib.pyplot as plt

# Load the dataset from a CSV file
data = pd.read_csv('../data/augmented_dataset.csv')

# Define features and target
X = data[['Application_Type', 'Signal_Strength', 'Latency', 'Required_Bandwidth', 'Allocated_Bandwidth']]
y = data['Resource_Allocation']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Load the model parameters from the CSV file
params_df = pd.read_csv('../data/light_gbm_best_params.csv')  # Ensure the correct file path and format

# Select the desired model parameters by model name (e.g., 'LightGBM_BO_GP')
model_name = 'LightGBM_BO_GP'
best_params = params_df[params_df['Model Name'] == model_name].iloc[0]

# Create the LightGBM model with the best parameters
lgb_model = lgb.LGBMRegressor(
    num_leaves=int(best_params['num_leaves']),
    n_estimators=int(best_params['n_estimators']),
    learning_rate=best_params['learning_rate'],
    max_depth=int(best_params['max_depth']),
    min_data_in_leaf=int(best_params['min_data_in_leaf']),
    lambda_l1=best_params['lambda_l1'],
    lambda_l2=best_params['lambda_l2'],
    random_state=42
)

# Fit the model to the training data
lgb_model.fit(X_train, y_train)

# Predictions on the training set
y_train_pred = lgb_model.predict(X_train)
# Predictions on the testing set
y_test_pred = lgb_model.predict(X_test)

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, mae, r2

# Calculate metrics for training and testing sets
train_metrics = calculate_metrics(y_train, y_train_pred)
test_metrics = calculate_metrics(y_test, y_test_pred)

# Print the metrics
print("Training Metrics:")
print(f"MSE: {train_metrics[0]:.4f}, MAE: {train_metrics[1]:.4f}, R2: {train_metrics[2]:.4f}")
print("Testing Metrics:")
print(f"MSE: {test_metrics[0]:.4f}, MAE: {test_metrics[1]:.4f}, R2: {test_metrics[2]:.4f}")

# Create a DataFrame for easy plotting
metrics_names = ['MSE', 'MAE', 'R2']
train_values = list(train_metrics)
test_values = list(test_metrics)

# Create a DataFrame for easy plotting
metrics_df = pd.DataFrame({
    'Metrics': metrics_names,
    'Training': train_values,
    'Testing': test_values
})

# Plotting the metrics differences
plt.figure(figsize=(12, 6))
bar_width = 0.35
x = np.arange(len(metrics_df['Metrics']))

# Bar for training metrics
plt.bar(x - bar_width/2, metrics_df['Training'], width=bar_width, label='Training Metrics', color='#4c72b0')
# Bar for testing metrics
plt.bar(x + bar_width/2, metrics_df['Testing'], width=bar_width, label='Testing Metrics', color='lightcoral')

# Add annotations to indicate the difference
for i in range(len(metrics_df)):
    difference = metrics_df['Testing'][i] - metrics_df['Training'][i]
    plt.text(x[i], max(metrics_df['Training'][i], metrics_df['Testing'][i]) + 0.05, 
             f"Diff: {difference:.2f}", ha='center', fontsize=10, color='black')

plt.xlabel('Metrics')
plt.ylabel('Values')
plt.xticks(x, metrics_df['Metrics'])
plt.legend()

plt.ylim(0, 1.8)  
plt.title("Training vs Testing Metrics for LightGBM")
plt.tight_layout()

# Save the plot
plt.savefig('../graphs/search_space/light_gbm_train_vs_test(new).png')

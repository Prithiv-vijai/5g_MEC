import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
import matplotlib.pyplot as plt

# Load the dataset from a CSV file
data = pd.read_csv('../data/augmented_dataset.csv')

# Define features and target
X = data[['Application_Type', 'Signal_Strength', 'Latency', 'Required_Bandwidth', 'Allocated_Bandwidth']]
y = data['Resource_Allocation']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model with the best parameters
best_params_tpe = {
    'l2_regularization': 3.114771316160287,
    'learning_rate': 0.09275685977836234,
    'max_depth': 18,
    'max_iter': 291,
    'max_leaf_nodes': 45,
    'min_samples_leaf': 13
}
# Create the model with the best parameters
model_tpe = HistGradientBoostingRegressor(**best_params_tpe, random_state=40)

# Fit the model to the training data
model_tpe.fit(X_train, y_train)

# Predictions on the training set
y_train_pred = model_tpe.predict(X_train)
# Predictions on the testing set
y_test_pred = model_tpe.predict(X_test)

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # MAPE calculation
    return mse, rmse, r2, mae, mape

# Calculate metrics for training and testing sets
train_metrics = calculate_metrics(y_train, y_train_pred)
test_metrics = calculate_metrics(y_test, y_test_pred)

# Create a DataFrame for easy plotting
metrics_names = ['MSE', 'RMSE', 'R2', 'MAE', 'MAPE']
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




plt.xlabel('Metrics')
plt.ylabel('Values')
plt.xticks(x, metrics_df['Metrics'])
plt.legend()



# Adjust text annotations to be inside the plot
for i in range(len(metrics_df)):
    # Position the text slightly above the bar but within the plot area
    plt.text(x[i] - bar_width/2, metrics_df['Training'][i] + 0.1, f"{metrics_df['Training'][i]:.2f}", ha='center', color='blue', fontsize=10)
    plt.text(x[i] + bar_width/2, metrics_df['Testing'][i] + 0.1, f"{metrics_df['Testing'][i]:.2f}", ha='center', color='red', fontsize=10)

plt.ylim(0, 1.8)  
plt.savefig('../graphs/model_output/train_vs_test.png')


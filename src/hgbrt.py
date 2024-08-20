import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error

# Load the dataset from a CSV file
data = pd.read_csv('../data/augmented_datasett.csv')

# Define features and target
X = data[['Application_Type', 'Signal_Strength', 'Latency', 'Required_Bandwidth', 'Allocated_Bandwidth']]
y = data['Resource_Allocation']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the default model
default_model = HistGradientBoostingRegressor()

# Fit the model
default_model.fit(X_train, y_train)

# Make predictions
y_pred_default = default_model.predict(X_test)

# Calculate metrics
def calculate_metrics(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    adjusted_r2 = 1 - (1-r2) * (len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, rmse, r2, adjusted_r2, mape, mae

metrics_default = calculate_metrics(y_test, y_pred_default)

# Print the metrics
metric_names = ['MSE', 'RMSE', 'R²', 'Adjusted R²', 'MAPE', 'MAE']
for name, value in zip(metric_names, metrics_default):
    print(f'{name}: {value:.4f}')

# Plot the metrics
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(metric_names, metrics_default, color='blue')
ax.set_title('Model Metrics without Parameter Optimization')
ax.set_ylabel('Metric Value')

# Display values on top of each bar
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}', ha='center', va='bottom', fontsize=10, weight='bold')

# Save the plot as a PNG file
plt.tight_layout()
plt.savefig('../graphs/model_output/hgbrt_no_optimization.png')

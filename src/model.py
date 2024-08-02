import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create the directory if it doesn't exist
output_dir = '../graphs/model_output'
os.makedirs(output_dir, exist_ok=True)

# Load the dataset from a CSV file
df = pd.read_csv('../data/preprocessed_augmented_dataset.csv')

# Define features (X) and target (y)
X = df[['Application_Type', 'Signal_Strength', 'Latency', 'Required_Bandwidth', 'Allocated_Bandwidth']]
y = df['Efficiency']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

# Dictionary to store the results
results = {}
predictions = {}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'MSE': mse, 'MAE': mae, 'R2': r2}
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

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.barplot(x='Model', y='MSE', data=metrics_df, ax=axes[0])
axes[0].set_title('MSE Comparison')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)

sns.barplot(x='Model', y='MAE', data=metrics_df, ax=axes[1])
axes[1].set_title('MAE Comparison')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)

sns.barplot(x='Model', y='R2', data=metrics_df, ax=axes[2])
axes[2].set_title('R2 Comparison')
axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'))
plt.show()

# Visualize the predicted vs actual values
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for i, (name, y_pred) in enumerate(predictions.items()):
    ax = axes[i]
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_title(f'{name} Predicted vs Actual')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'predicted_vs_actual.png'))
plt.show()

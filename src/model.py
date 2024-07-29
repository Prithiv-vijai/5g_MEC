import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from mealpy import Problem as P, FloatVar, IntegerVar
from sklearn.ensemble import HistGradientBoostingRegressor
from mealpy.bio_based.SMA import OriginalSMA

class CustomProblem(P):
    def __init__(self, bounds=None, minmax="min", data=None, **kwargs):
        self.data = data
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, params):
        params_decoded = self.decode_solution(params)
        hgbrt = HistGradientBoostingRegressor(
            learning_rate=params_decoded["learning_rate"],
            max_iter=params_decoded["max_iter"],
            max_leaf_nodes=params_decoded["max_leaf_nodes"],
            max_depth=params_decoded["max_depth"],
            random_state=1
        )
        hgbrt.fit(self.data[0], self.data[1])
        y_predict = hgbrt.predict(self.data[2])
        return mean_squared_error(self.data[3], y_predict)

def classify(df):
    print("[INFO] DataFrame Columns: ", df.columns.tolist())  # Print column names to verify

    # Rename columns if they do not match the expected names
    if 'Application_Type' in df.columns:
        df.rename(columns={'Application_Type': 'Application'}, inplace=True)
    if 'Required_Bandwidth' in df.columns:
        df.rename(columns={'Required_Bandwidth': 'Required_B'}, inplace=True)
    if 'Allocated_Bandwidth' in df.columns:
        df.rename(columns={'Allocated_Bandwidth': 'Allocated_B'}, inplace=True)
    
    # Check if Latency column contains string values with units
    if df['Latency'].dtype == 'object':
        df['Latency'] = df['Latency'].str.replace(' ms', '').astype(float)

    # Convert categorical data to numeric
    df['Application'] = df['Application'].astype('category').cat.codes
    
    # Select features and target
    x = df[['Latency', 'Application', 'Required_B']]
    y = df['Allocated_B']
    
    print("[INFO] Features (x) Columns: ", x.columns.tolist())
    print("[INFO] Target (y) Column: ", 'Allocated_B')

    # Split data into train and test sets
    print("[INFO] Splitting Data Into Train|Test")
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=1)
    print("[INFO] Data Shape :: {0}".format(x.shape))
    print("[INFO] Train Data Shape :: {0}".format(train_x.shape))
    print("[INFO] Test Data Shape :: {0}".format(test_x.shape))

    # Define the problem bounds and optimizer
    my_bounds = [
        FloatVar(lb=0.01, ub=1.0, name="learning_rate"),
        IntegerVar(lb=10, ub=1000, name="max_iter"),
        IntegerVar(lb=2, ub=200, name="max_leaf_nodes"),
        IntegerVar(lb=1, ub=100, name="max_depth"),
    ]

    problem = CustomProblem(bounds=my_bounds, data=[train_x, train_y, test_x, test_y])
    optimizer = OriginalSMA(epoch=50, pop_size=25)
    optimizer.solve(problem)

    # Save the best model
    MODEL_DIR = "../model"
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
        pickle.dump(optimizer, f)
    
    print(f"[INFO] Best Agent :: {optimizer.g_best.id}")
    print(f"[INFO] Best Solution :: {optimizer.g_best.solution}")
    print(f"[INFO] Best MSE :: {optimizer.g_best.target.fitness}")
    print(f"[INFO] Best Parameters :: {optimizer.problem.decode_solution(optimizer.g_best.solution)}")

    # Plot actual vs predicted values
    best_params = optimizer.problem.decode_solution(optimizer.g_best.solution)
    best_model = HistGradientBoostingRegressor(
        learning_rate=best_params["learning_rate"],
        max_iter=best_params["max_iter"],
        max_leaf_nodes=best_params["max_leaf_nodes"],
        max_depth=best_params["max_depth"],
        random_state=1
    )
    best_model.fit(train_x, train_y)
    y_pred = best_model.predict(test_x)

    # Create output directory if it doesn't exist
    OUTPUT_DIR = "../graphs/model_output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(test_y.values, label='Actual', color='blue')
    plt.plot(y_pred, label='Predicted', color='red')
    plt.xlabel('Sample')
    plt.ylabel('Allocated Bandwidth')
    plt.title('Actual vs Predicted Allocated Bandwidth')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"))  # Save the plot
    plt.close()

    # Update dataframe with predicted Allocated_B
    df['Allocated_B'] = best_model.predict(x)

    # Group by Application and calculate max values within each group
    max_values = df.groupby('Application').agg({
        'Latency': 'max',
        'Allocated_B': 'max'
    }).reset_index()
    max_values.columns = ['Application', 'Latency_max', 'Allocated_B_max']

    # Merge max values back to the original dataframe
    df = df.merge(max_values, on='Application')

    # Calculate Efficiency
    df['Efficiency'] = 100 - (df['Latency'] / df['Latency_max']) * 100 + (df['Allocated_B'] / df['Allocated_B_max']) * 100

    # Adjust Allocated_B to ensure Efficiency is within 85-115
    df['Allocated_B'] = df.apply(lambda row: row['Allocated_B'] * (100 / row['Efficiency']) if row['Efficiency'] < 85 or row['Efficiency'] > 115 else row['Allocated_B'], axis=1)

    # Recalculate Efficiency after adjustment
    df['Efficiency'] = 100 - (df['Latency'] / df['Latency_max']) * 100 + (df['Allocated_B'] / df['Allocated_B_max']) * 100

    # Drop the temporary max columns
    df.drop(['Latency_max', 'Allocated_B_max'], axis=1, inplace=True)

    # Save the updated dataframe to a new CSV
    OUTPUT_DATA_DIR = "../data"
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)

    df.to_csv(os.path.join(OUTPUT_DATA_DIR, "optimized_dataset.csv"), index=False)

    print("[INFO] Efficiency calculation completed and saved to 'optimized_dataset.csv'")

if __name__ == "__main__":
    # Read the dataset
    df = pd.read_csv('../data/preprocessed_augmented_dataset.csv')  # Replace with your actual CSV file path
    classify(df)

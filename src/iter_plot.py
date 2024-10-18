import matplotlib.pyplot as plt

# Data
iterations = [100, 200, 300, 400, 500, 750, 1000]

# MSE values for different methods
mse_values_tpe = [
    1.1972900340542136,
    1.1196773658200143,
    1.023528992393525,
    1.0129161104144256,
    1.010244609622626,
    1.010107006599946,
    1.0100134152746776
]

mse_values_gp = [
    1.1085,
    1.0031,
    1.0007,
    1.0006,
    1.0006,
    1.0006,
    1.0006
]

mse_values_cmaes = [
    1.2203,
    1.1825,
    1.0961,
    1.0443,
    1.0113,
    1.0113,
    1.0113
]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(iterations, mse_values_tpe, marker='o', color='#4c72b0', label='TPE', alpha=0.7)
plt.plot(iterations, mse_values_gp, marker='s', color='#55a868', label='GP ', alpha=0.7)
plt.plot(iterations, mse_values_cmaes, marker='d', color='#c44e52', label='CMAES ', alpha=0.7)

# Labels and ticks
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error (MSE)')
plt.xticks(iterations)
plt.yticks([1.25,1.20, 1.15, 1.10, 1.05, 1.00, 0.95])

# Grid, legend, and layout settings
plt.grid()
plt.legend()
plt.ylim(0.95, 1.25)
plt.tight_layout()

# Save the plot
plt.savefig('../graphs/model_output/convergence_comparison.png')

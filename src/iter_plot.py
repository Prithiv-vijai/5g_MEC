import matplotlib.pyplot as plt

# Data
iterations = [50, 100, 150, 200, 300, 400, 500, 1000]
mse_values = [
    1.1172900340542136,
    1.1096773658200143,
    1.103528992393525,
    1.1029161104144256,
    1.08289685489361,
    1.065407006599946,  # Convergence point
    1.065407006599946,
    1.065407006599946
]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(iterations, mse_values, marker='o', color='#4c72b0', label='MSE Values')
plt.axvline(x=400, color='r', linestyle='--', label='Convergence Point (Iterations = 400)')


plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error (MSE)')

# Set x-ticks to be the iteration points
plt.xticks(iterations)

# Set y-ticks with adjusted positions for better spacing
y_ticks = [1.12, 1.11, 1.10, 1.09, 1.08, 1.07, 1.06]  # Custom y-tick values
plt.yticks(y_ticks)

plt.grid()
plt.legend()
plt.ylim(1.05, 1.12)  # Set y-limits for better visualization
plt.tight_layout()

# Save the plot
plt.savefig('../graphs/model_output/tpe_convergence.png')

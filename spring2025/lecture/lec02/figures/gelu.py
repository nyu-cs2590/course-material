import numpy as np
import matplotlib.pyplot as plt
from math import erf, sqrt

def gelu(x):
    """Gaussian Error Linear Unit (GELU)."""
    return 0.5 * x * (1 + erf(x / sqrt(2)))

# Generate a range of x values
x_vals = np.linspace(-5, 5, 1000)

# Compute GELU for each x
y_vals = [gelu(x) for x in x_vals]

# Create the plot
plt.figure(figsize=(6, 4))
plt.plot(x_vals, y_vals, label='GELU(x)', color='blue')
plt.axhline(0, color='black', linewidth=0.5)  # Horizontal axis for reference
plt.axvline(0, color='black', linewidth=0.5)  # Vertical axis for reference

# Add labels and title
plt.title('GELU Activation Function')
plt.xlabel('x')
plt.ylabel('GELU(x)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left')

# Display the plot
plt.savefig('gelu.pdf')


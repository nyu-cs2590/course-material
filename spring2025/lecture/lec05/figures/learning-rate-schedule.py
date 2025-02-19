import numpy as np
import matplotlib.pyplot as plt

# Parameters: you can adjust these values as needed
total_steps = 10000      # Total training steps
warmup_steps = 1000      # Steps for the warmup phase
base_lr = 1.0            # Base learning rate after warmup

# Create an array of training steps
steps = np.arange(total_steps)

# Pre-allocate arrays for the learning rates
lr_linear = np.zeros_like(steps, dtype=float)
lr_cosine = np.zeros_like(steps, dtype=float)

# Compute learning rate for each step for the Linear Decay schedule
for i, step in enumerate(steps):
    if step < warmup_steps:
        # Warmup: linearly increase lr from 0 to base_lr
        lr = base_lr * (step / warmup_steps)
    else:
        # Linear decay: decay linearly from base_lr to 0 over the remaining steps
        lr = base_lr * (1 - (step - warmup_steps) / (total_steps - warmup_steps))
        lr = max(lr, 0.0)  # Ensure the learning rate doesn't go negative
    lr_linear[i] = lr

# Compute learning rate for each step for the Cosine Decay schedule
for i, step in enumerate(steps):
    if step < warmup_steps:
        # Warmup: same as above
        lr = base_lr * (step / warmup_steps)
    else:
        # Cosine decay: decay following a half-cosine curve
        # Here, progress goes from 0 to 1 over the steps after warmup.
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        lr = base_lr * 0.5 * (1 + np.cos(np.pi * progress))
    lr_cosine[i] = lr

# Plotting both schedules side by side
plt.figure(figsize=(12, 5))

# Plot for Linear Decay
plt.subplot(1, 2, 1)
plt.plot(steps, lr_linear, color='blue')
plt.title('Linear Decay with Warmup')
plt.xlabel('Training Step')
plt.ylabel('Learning Rate')
plt.grid(True)

# Plot for Cosine Decay
plt.subplot(1, 2, 2)
plt.plot(steps, lr_cosine, color='green')
plt.title('Cosine Decay with Warmup')
plt.xlabel('Training Step')
plt.ylabel('Learning Rate')
plt.grid(True)

plt.tight_layout()
plt.savefig('learning-rate.pdf')


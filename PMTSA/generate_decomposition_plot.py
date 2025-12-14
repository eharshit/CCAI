
import numpy as np
import matplotlib.pyplot as plt

# Generate time steps
t = np.linspace(0, 4 * np.pi, 100)

# Trend: Linear increase
trend = np.linspace(10, 30, len(t))

# Seasonality: Sine wave
seasonality = 5 * np.sin(3 * t)

# Additive Model: Y = Trend + Seasonality
additive = trend + seasonality

# Multiplicative Model: Y = Trend * (1 + Seasonality_scaled)
# Scaling seasonality to be a percentage fluctuation for multiplicative
seasonality_multiplicative = 0.2 * np.sin(3 * t) 
multiplicative = trend * (1 + seasonality_multiplicative)

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot Additive
ax1.plot(t, additive, label='Observed', color='blue', linewidth=2)
ax1.plot(t, trend, label='Trend', color='red', linestyle='--', alpha=0.7)
ax1.set_title('Additive Decomposition\n(Constant Seasonality)', fontsize=14)
ax1.set_xlabel('Time')
ax1.set_ylabel('Value')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

# Plot Multiplicative
ax2.plot(t, multiplicative, label='Observed', color='green', linewidth=2)
ax2.plot(t, trend, label='Trend', color='red', linestyle='--', alpha=0.7)
ax2.set_title('Multiplicative Decomposition\n(Seasonality Scales with Trend)', fontsize=14)
ax2.set_xlabel('Time')
ax2.set_ylabel('Value')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('c:\\Users\\harsh\\Desktop\\CCAI-1\\PMTSA\\decomposition_comparison.png', dpi=300)
print("Image saved successfully.")

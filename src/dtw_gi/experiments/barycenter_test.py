import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from data.data_generation import make_spirals, plot_trajectory, get_rot2d
from dtw_gi.barycenters import SoftDTWGIBarycenter

# 1. Setup Data: Generate 5 spirals with random rotations
np.random.seed(42)

n_series = 5
base_dataset = make_spirals(n=n_series, sz=100, noise=0.02)
dataset = []

print("Generating dataset with random rotations...")
for i in range(n_series):
    # Convert to torch
    ts = base_dataset[i]
    # Apply a random rotation to make it "heterogeneous" in space
    angle = np.random.uniform(0, 2 * np.pi)
    dataset.append(np.dot(ts, get_rot2d(angle)))

# 2. Run softDTW-GI Barycenter
# We want a barycenter of length 100 and dimension 2
bary_model = SoftDTWGIBarycenter(T=100, p=2, gamma=0.1, max_iter=100, lr_b=0.05)
barycenter, transforms = bary_model.fit(dataset, verbose=True)

# 3. Visualization
plt.figure(figsize=(12, 5))

# Plot all original (rotated) series
plt.subplot(1, 2, 1)
for i in range(n_series):
    ts_np = dataset[i]
    plt.plot(ts_np[:, 0], ts_np[:, 1], alpha=0.4, label=f"Series {i}")
plt.title("Original Rotated Series")
plt.axis("equal")

# Plot the recovered barycenter
plt.subplot(1, 2, 2)
bary_np = barycenter
plt.plot(bary_np[:, 0], bary_np[:, 1], color='red', linewidth=3, label="Barycenter")
plt.title("Computed DTW-GI Barycenter")
plt.axis("equal")
plt.legend()

plt.tight_layout()
plt.show()
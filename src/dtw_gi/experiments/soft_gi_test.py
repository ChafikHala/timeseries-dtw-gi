import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import matplotlib.pyplot as plt
from data.data_generation import make_spirals, plot_trajectory
from dtw_gi.softdtw_gi import soft_dtw_gi

dataset = make_spirals(n=2, sz=100, noise=0.01)
x, y = dataset[0], dataset[1]

# Run softDTW-GI
res = soft_dtw_gi(x, y, gamma=0.1, max_iter=150, lr=0.05, verbose=True)

# Visualize
y_aligned = (y @ res.P.T)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(x[:,0], x[:,1], label='Target (x)')
plt.plot(y[:,0], y[:,1], label='Original (y)', alpha=0.5)
plt.title("Before Alignment")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x[:,0], x[:,1], label='Target (x)')
plt.plot(y_aligned[:,0], y_aligned[:,1], label='Aligned (y_hat)', color='green')
plt.title("After softDTW-GI")
plt.legend()
plt.show()
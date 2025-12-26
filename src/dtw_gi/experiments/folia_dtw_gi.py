import numpy as np
import torch
import matplotlib.pyplot as plt

from dtw_gi.dtw.backend import compute_dtw_path
from dtw_gi.bcd import dtw_gi

from data.data_generation import (
    make_folia,
    plot_trajectory,
    set_fig_style,
)

# 1. Generate folia data
np.random.seed(0)

set_fig_style(font_size=18)

dataset = make_folia(
    n=2,
    sz=140,
    noise=0.02,
    shift=False,
    some_3d=False,
)

x, y = dataset[0], dataset[1]

print("x:", x.shape)
print("y:", y.shape)



# 2. Baseline DTW
path_dtw, cost_dtw = compute_dtw_path(x, y)
print(f"DTW cost (no GI): {cost_dtw:.4f}")


# 3. DTW-GI (BCD + Stiefel)
res = dtw_gi(
    x,
    y,
    max_iter=50,
    verbose=True,
)

P_hat = res.P
path_gi = res.path
cost_gi = res.cost

print(f"DTW-GI cost: {cost_gi:.4f}")
y_aligned = y @ P_hat.T


# 4. Visualization

fig = plt.figure(figsize=(15, 4))

# (a) Original folia
ax1 = fig.add_subplot(131)
plot_trajectory(x, ax1, color_code="tab:blue")
plot_trajectory(y, ax1, color_code="tab:orange", alpha=0.7)
ax1.set_title("Original folia\n(rotated)")
ax1.axis("equal")

# (b) After DTW-GI
ax2 = fig.add_subplot(132)
plot_trajectory(x, ax2, color_code="tab:blue")
plot_trajectory(y_aligned, ax2, color_code="tab:green", alpha=0.7)
ax2.set_title("After DTW-GI")
ax2.axis("equal")

# (c) Alignment path
ax3 = fig.add_subplot(133)
px = [i for i, j in path_gi]
py = [j for i, j in path_gi]
ax3.plot(px, py)
ax3.set_xlabel("x index")
ax3.set_ylabel("y index")
ax3.set_title("DTW-GI alignment path")

plt.tight_layout()
plt.show()


fig = plt.figure(figsize=(15, 8))

# DTW (no GI)
# Geometry
ax1 = fig.add_subplot(231)
plot_trajectory(x, ax1, color_code="tab:blue")
plot_trajectory(y, ax1, color_code="tab:orange", alpha=0.7)
ax1.set_title("DTW (no GI): geometry")
ax1.axis("equal")

# Alignment path
ax2 = fig.add_subplot(232)
px = [i for i, j in path_dtw]
py = [j for i, j in path_dtw]
ax2.plot(px, py)
ax2.set_title("DTW (no GI): alignment path")
ax2.set_xlabel("x index")
ax2.set_ylabel("y index")



plt.tight_layout()
plt.show()


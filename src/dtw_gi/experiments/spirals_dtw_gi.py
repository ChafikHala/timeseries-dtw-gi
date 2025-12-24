import numpy as np
import torch
import matplotlib.pyplot as plt

from dtw_gi.dtw.backend import compute_dtw_path
from dtw_gi.dtw.alignment import path_to_W
from dtw_gi.bcd import dtw_gi_bcd_stiefel

from data.data_generation import (
    make_spirals,
    plot_trajectory,
    set_fig_style,
)


# 1. Generate data

np.random.seed(0)
torch.manual_seed(0)

set_fig_style(font_size=18)

dataset = make_spirals(
    n=2,
    sz=120,
    noise=0.02,
    shift=False,
    some_3d=False,
)

x_np, y_np = dataset[0], dataset[1]

# Convert to torch
x = torch.tensor(x_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32)

print("x:", x.shape)
print("y:", y.shape)

# 2. Baseline DTW (no GI)
path_dtw, cost_dtw = compute_dtw_path(x, y, backend="tslearn")
print(f"DTW cost (no GI): {cost_dtw:.4f}")



# 3. DTW-GI (BCD + Stiefel)
res = dtw_gi_bcd_stiefel(
    x,
    y,
    max_iter=40,
    tol=1e-6,
    backend="torch",
    verbose=True,
)

P_hat = res.P
path_gi = res.path
cost_gi = res.cost

print(f"DTW-GI cost: {cost_gi:.4f}")
print("Recovered P:")
print(P_hat)
print("P^T P ≈ I:")
print(P_hat.T @ P_hat)

y_aligned = y @ P_hat.T


# 4. Visualization
fig = plt.figure(figsize=(15, 4))

# (a) Original spirals
ax1 = fig.add_subplot(131)
plot_trajectory(x_np, ax1, color_code="tab:blue")
plot_trajectory(y_np, ax1, color_code="tab:orange", alpha=0.7)
ax1.set_title("Original spirals\n(rotation + time warp)")
ax1.axis("equal")

# (b) After DTW-GI
ax2 = fig.add_subplot(132)
plot_trajectory(x_np, ax2, color_code="tab:blue")
plot_trajectory(y_aligned.detach().numpy(), ax2, color_code="tab:green", alpha=0.7)
ax2.set_title("After DTW-GI\n(global rotation removed)")
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

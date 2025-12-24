import numpy as np
import torch
import matplotlib.pyplot as plt

from dtw_gi.dtw.backend import compute_dtw_path
from dtw_gi.bcd import dtw_gi_bcd_stiefel, dtw_gi_bcd_stiefel_multistart

from data.data_generation import make_one_folium, get_rot2d, set_fig_style


# Configuration

T = 140
noise_std = 0.01
angles = np.linspace(0.0, 2*np.pi, 15)
device = "cpu"

torch.manual_seed(0)
np.random.seed(0)

set_fig_style(font_size=18)


# Reference trajectory x
x_np = make_one_folium(sz=T, noise=0.0)
x = torch.tensor(x_np, dtype=torch.float32, device=device)

# Baseline costs (alpha = 0)
path0, cost_dtw_0 = compute_dtw_path(x, x, backend="torch")

res0 = dtw_gi_bcd_stiefel(
    x,
    x,
    max_iter=30,
    backend="torch",
)
cost_gi_0 = res0.cost


# Sweep over rotations

dtw_ratios = []
gi_ratios = []

for alpha in angles:
    R = torch.tensor(
        get_rot2d(alpha),
        dtype=torch.float32,
        device=device,
    )

    y = (x @ R.T) + noise_std * torch.randn_like(x)

    # --- DTW ---
    path_dtw, cost_dtw = compute_dtw_path(x, y, backend="torch")
    dtw_ratios.append(cost_dtw / len(path_dtw))

    # --- DTW-GI ---
    res = dtw_gi_bcd_stiefel_multistart(
            x,
            y,
            max_iter=30,
            n_random_starts=5,
            seed=0,
            backend="torch",
            verbose=False,
    )
    gi_ratios.append(res.cost / len(res.path))


    print(
        f"alpha={alpha:.2f} | "
        f"DTW={cost_dtw:.3f} | "
        f"DTW-GI={res.cost:.3f}"
    )



plt.figure(figsize=(6, 4))
plt.plot(angles, dtw_ratios, "-o", label="DTW", linewidth=2)
plt.plot(angles, gi_ratios, "-o", label="DTW-GI", linewidth=2)

plt.xlabel(r"Rotation angle $\alpha$")
plt.ylabel("Normalized distance")
plt.title("Rotation invariance of DTW vs DTW-GI")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

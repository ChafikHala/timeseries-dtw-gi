from __future__ import annotations

from typing import List, Tuple, Literal
import torch
import numpy as np

from .dtw import dtw


DTWBackend = Literal["torch", "tslearn"]


def compute_dtw_path(
    x: torch.Tensor,
    y: torch.Tensor,
    backend: DTWBackend = "torch",
) -> Tuple[List[Tuple[int, int]], float]:
    """
    Unified DTW path interface for BCD.

    Args:
        x: (Tx, p) torch tensor
        y: (Ty, p) torch tensor
        backend: "torch" | "tslearn"

    Returns:
        path: list of (i, j)
        cost: float
    """
    if backend == "torch":
        res = dtw(x, y, return_path=True)
        if res.path is None:
            raise RuntimeError("Torch DTW did not return a path.")
        return res.path, float(res.cost)

    elif backend == "tslearn":
        return _dtw_path_tslearn(x, y)

    else:
        raise ValueError(f"Unknown DTW backend: {backend}")
    


def _dtw_path_tslearn(
    x: torch.Tensor,
    y: torch.Tensor,
):
    try:
        from tslearn.metrics import dtw_path
    except ImportError as e:
        raise ImportError(
            "tslearn is required for backend='tslearn'. "
            "Install with `pip install tslearn`."
        ) from e

    # Convert to numpy ONCE
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    path, _ = dtw_path(x_np, y_np)

    # Squared Euclidean cost (to match Torch DTW definition)
    cost_sq = sum(
        np.sum((x_np[i] - y_np[j]) ** 2)
        for i, j in path
    )

    path = [(int(i), int(j)) for i, j in path]
    return path, float(cost_sq)



from __future__ import annotations

from typing import List, Tuple, Literal
import numpy as np

DTWBackend = Literal["torch", "tslearn"]


def compute_dtw_path(
    x: np.ndarray,
    y: np.ndarray,
    backend: DTWBackend = "tslearn",
) -> Tuple[List[Tuple[int, int]], float]:
    """
    Unified DTW path interface.

    Args:
        x: (Tx, p) numpy array
        y: (Ty, p) numpy array
        backend: "tslearn" | "torch"

    Returns:
        path: list of (i, j)
        cost: float (sum of squared Euclidean distances along the path)
    """
    if backend == "tslearn":
        return _dtw_path_tslearn(x, y)

    elif backend == "torch":
        return _dtw_path_torch(x, y)

    else:
        raise ValueError(f"Unknown DTW backend: {backend}")


def _dtw_path_tslearn(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[List[Tuple[int, int]], float]:
    try:
        from tslearn.metrics import dtw_path
    except ImportError as e:
        raise ImportError(
            "tslearn is required for backend='tslearn'. "
            "Install with `pip install tslearn`."
        ) from e

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    path, _ = dtw_path(x, y)

    cost = sum(
        np.sum((x[i] - y[j]) ** 2)
        for i, j in path
    )

    path = [(int(i), int(j)) for i, j in path]
    return path, float(cost)


def _dtw_path_torch(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[List[Tuple[int, int]], float]:
    import torch
    from .dtw import dtw  

    x_t = torch.as_tensor(x, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32)

    res = dtw(x_t, y_t, return_path=True)
    if res.path is None:
        raise RuntimeError("Torch DTW did not return a path.")

    path = [(int(i), int(j)) for i, j in res.path]
    cost = float(res.cost)

    return path, cost

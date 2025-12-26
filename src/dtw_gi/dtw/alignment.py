from __future__ import annotations

from typing import Iterable, List, Tuple
import numpy as np


def path_to_W(
    path: Iterable[Tuple[int, int]],
    Tx: int,
    Ty: int,
) -> np.ndarray:
    """
    Build alignment matrix Wπ ∈ {0,1}^{Tx×Ty} from a DTW warping path.

    This is the exact equivalent of the authors' `path2mat`.
    """
    W = np.zeros((Tx, Ty), dtype=np.float64)
    for i, j in path:
        W[i, j] = 1.0
    return W


def W_to_path(W: np.ndarray) -> List[Tuple[int, int]]:
    """
    Extract (i, j) indices where W[i, j] == 1.
    """
    if W.ndim != 2:
        raise ValueError("W must be a 2D array.")

    idx = np.argwhere(W > 0.0)
    return [(int(i), int(j)) for i, j in idx]

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch


@dataclass(frozen=True)
class DTWResult:
    cost: torch.Tensor                 # scalar
    D: torch.Tensor                    # accumulated cost matrix (Tx+1, Ty+1)
    path: Optional[List[Tuple[int,int]]]  # (i,j) in 0-based time indices


def _pairwise_sqeuclidean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    x: (Tx, p), y: (Ty, p)
    returns C: (Tx, Ty) where C[i,j] = ||x_i - y_j||^2
    """
    # (Tx,1,p) - (1,Ty,p) -> (Tx,Ty,p)
    diff = x[:, None, :] - y[None, :, :]
    return (diff * diff).sum(dim=-1)


def dtw(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    return_path: bool = True,
) -> DTWResult:
    """
    Classic DTW via dynamic programming.

    Args:
      x: (Tx, p) float tensor
      y: (Ty, p) float tensor

    Returns:
      DTWResult with:
        - cost: D[Tx,Ty]
        - D: accumulated matrix with padding row/col
        - path: optimal warping path (if return_path)
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be (T, p) tensors.")
    if x.size(1) != y.size(1):
        raise ValueError("DTW requires same feature dimension p for x and y.")

    Tx, _ = x.shape
    Ty, _ = y.shape
    device = x.device
    dtype = x.dtype

    C = _pairwise_sqeuclidean(x, y)  # (Tx, Ty)

    inf = torch.tensor(torch.inf, device=device, dtype=dtype)
    D = torch.full((Tx + 1, Ty + 1), inf, device=device, dtype=dtype)
    D[0, 0] = 0.0

    # DP recurrence:
    # D[i,j] = C[i-1,j-1] + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    for i in range(1, Tx + 1):
        # small speedup: local variable
        Di = D[i]
        Dim1 = D[i - 1]
        for j in range(1, Ty + 1):
            best_prev = torch.minimum(
                torch.minimum(Dim1[j], Di[j - 1]),
                Dim1[j - 1],
            )
            Di[j] = C[i - 1, j - 1] + best_prev

    if not return_path:
        return DTWResult(cost=D[Tx, Ty], D=D, path=None)

    # Backtrace from (Tx,Ty) to (0,0)
    i, j = Tx, Ty
    path: List[Tuple[int, int]] = []
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            path.append((i - 1, j - 1))
        # choose argmin among predecessors (with boundary checks)
        candidates = []
        if i > 0:
            candidates.append((D[i - 1, j], i - 1, j))
        if j > 0:
            candidates.append((D[i, j - 1], i, j - 1))
        if i > 0 and j > 0:
            candidates.append((D[i - 1, j - 1], i - 1, j - 1))
        _, i, j = min(candidates, key=lambda t: float(t[0]))

    path.reverse()
    return DTWResult(cost=D[Tx, Ty], D=D, path=path)

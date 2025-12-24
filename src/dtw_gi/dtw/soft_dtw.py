from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class SoftDTWResult:
    cost: torch.Tensor      # scalar
    D: torch.Tensor         # (Tx+1, Ty+1) accumulated matrix


def _pairwise_sqeuclidean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    x: (Tx, p)
    y: (Ty, p)
    returns: C (Tx, Ty) with C[i,j] = ||x_i - y_j||^2
    """
    diff = x[:, None, :] - y[None, :, :]
    return (diff * diff).sum(dim=-1)


def _softmin3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, gamma: float):
    """
    Numerically stable soft-min over three scalars.
    """
    stacked = torch.stack([a, b, c])
    return -gamma * torch.logsumexp(-stacked / gamma, dim=0)


def soft_dtw(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    gamma: float,
) -> SoftDTWResult:
    """
    Soft-DTW between two multivariate time series.

    Args:
      x: (Tx, p) tensor
      y: (Ty, p) tensor
      gamma: smoothing parameter (>0)

    Returns:
      SoftDTWResult:
        - cost: scalar soft-DTW value
        - D: full DP matrix (Tx+1, Ty+1)
    """
    if gamma <= 0:
        raise ValueError("gamma must be strictly positive for soft-DTW.")

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be (T, p) tensors.")

    if x.size(1) != y.size(1):
        raise ValueError("soft-DTW requires same feature dimension.")

    Tx, _ = x.shape
    Ty, _ = y.shape
    device = x.device
    dtype = x.dtype

    C = _pairwise_sqeuclidean(x, y)  # (Tx, Ty)

    inf = torch.tensor(float("inf"), device=device, dtype=dtype)
    D = torch.full((Tx + 1, Ty + 1), inf, device=device, dtype=dtype)
    D[0, 0] = 0.0

    # DP recursion with soft-min
    for i in range(1, Tx + 1):
        Di = D[i]
        Dim1 = D[i - 1]
        for j in range(1, Ty + 1):
            Di[j] = C[i - 1, j - 1] + _softmin3(
                Dim1[j],      # up
                Di[j - 1],    # left
                Dim1[j - 1],  # diagonal
                gamma,
            )

    return SoftDTWResult(cost=D[Tx, Ty], D=D)

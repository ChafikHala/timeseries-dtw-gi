# dtw_gi/stiefel.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import numpy as np


@dataclass(frozen=True)
class StiefelUpdateResult:
    P: torch.Tensor            # (px, py)
    M: torch.Tensor            # (px, py)  = x^T W y
    singular_values: torch.Tensor  # (min(px,py),)


def stiefel_procrustes_update(
    x: torch.Tensor,
    y: torch.Tensor,
    W: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> StiefelUpdateResult:
    """
    Given x in R^{Tx x px}, y in R^{Ty x py}, and alignment matrix W in {0,1}^{Tx x Ty},
    compute P* = argmax_{P^T P = I} <x^T W y, P>.

    This is solved by SVD of M = x^T W y: M = U S V^T, P = U V^T.

    Shapes:
        x: (Tx, px)
        y: (Ty, py)
        W: (Tx, Ty)
        returns P: (px, py)
    """
    if x.ndim != 2 or y.ndim != 2 or W.ndim != 2:
        raise ValueError("x, y, W must be 2D tensors.")
    Tx, px = x.shape
    Ty, py = y.shape
    if W.shape != (Tx, Ty):
        raise ValueError(f"W must have shape (Tx, Ty)=({Tx},{Ty}), got {tuple(W.shape)}.")

    # M = x^T W y  -> (px, Tx)(Tx, Ty)(Ty, py) = (px, py)
    M = x.transpose(0, 1) @ (W @ y)

    # SVD: M = U diag(S) Vh, with Vh = V^T
    # full_matrices=False gives U:(px,k), Vh:(k,py), k=min(px,py)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)

    # P = U V^T  -> (px,k)(k,py) = (px,py)
    P = U @ Vh

    # Numerical hygiene: enforce orthonormal columns (optional)
    # This projection is cheap and stabilizes drift.
    # P <- P (P^T P)^{-1/2}
    G = P.transpose(0, 1) @ P  # (py, py)
    # symmetric eig for inverse sqrt
    evals, evecs = torch.linalg.eigh(G)
    evals = torch.clamp(evals, min=eps)
    inv_sqrt = evecs @ torch.diag(evals.rsqrt()) @ evecs.transpose(0, 1)
    P = P @ inv_sqrt

    return StiefelUpdateResult(P=P, M=M, singular_values=S)


def apply_linear_map(y: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """
    Apply f(y) = y P^T when y is (Ty, py) and P is (px, py).
    Output is (Ty, px).
    """
    if y.ndim != 2 or P.ndim != 2:
        raise ValueError("y and P must be 2D tensors.")
    px, py = P.shape
    if y.shape[1] != py:
        raise ValueError(f"y has feature dim {y.shape[1]}, expected py={py}.")
    return y @ P.transpose(0, 1)

def random_stiefel(px: int, py: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample a random matrix P ∈ V_{py,px} (columns orthonormal).
    """
    A = rng.standard_normal((px, py))
    Q, _ = np.linalg.qr(A)
    return Q[:, :py]



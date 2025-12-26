from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class StiefelUpdateResult:
    P: np.ndarray                 # (px, py)
    M: np.ndarray                 # (px, py)
    singular_values: np.ndarray   # (min(px, py),)


def stiefel_procrustes_update(
    x: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
) -> StiefelUpdateResult:
    """
    Procrustes update on the Stiefel manifold.

    Solves:
        P* = argmax_{P^T P = I} < x^T W y, P >

    via SVD:
        M = x^T W y = U S V^T
        P = U V^T

    Shapes:
        x: (Tx, px)
        y: (Ty, py)
        W: (Tx, Ty)

    Returns:
        P: (px, py)
    """
    if x.ndim != 2 or y.ndim != 2 or W.ndim != 2:
        raise ValueError("x, y, W must be 2D arrays.")

    Tx, px = x.shape
    Ty, py = y.shape

    if W.shape != (Tx, Ty):
        raise ValueError(
            f"W must have shape (Tx, Ty)=({Tx},{Ty}), got {W.shape}"
        )

    M = x.T @ W @ y        # (px, py)

    # SVD 
    U, S, Vt = np.linalg.svd(M, full_matrices=False)

    # Stiefel solution
    P = U @ Vt             # (px, py)

    return StiefelUpdateResult(P=P, M=M, singular_values=S)



def apply_linear_map(y: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Apply f(y) = y P^T

    y: (Ty, py)
    P: (px, py)
    returns: (Ty, px)
    """
    if y.ndim != 2 or P.ndim != 2:
        raise ValueError("y and P must be 2D arrays.")

    px, py = P.shape
    if y.shape[1] != py:
        raise ValueError(
            f"y has feature dim {y.shape[1]}, expected py={py}"
        )

    return y @ P.T


def random_stiefel(px: int, py: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample a random matrix P ∈ V_{py,px} (orthonormal columns).
    """
    A = rng.standard_normal((px, py))
    Q, _ = np.linalg.qr(A)
    return Q[:, :py]

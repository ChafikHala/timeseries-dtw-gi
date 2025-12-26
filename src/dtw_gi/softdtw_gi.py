from __future__ import annotations

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass

from tslearn.metrics import soft_dtw


# Utilities

def _project_to_stiefel(P: np.ndarray) -> np.ndarray:
    """
    Projection onto Stiefel manifold via SVD: P <- U V^T
    """
    U, _, Vt = np.linalg.svd(P, full_matrices=False)
    return U @ Vt


def _apply_linear_map(y: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    f(y) = y @ P^T
    """
    return y @ P.T


def _normalized_softdtw(x, y, gamma: float) -> float:
    """
    Normalized softDTW:
        softDTW(x, y)
        - 0.5 * (softDTW(x, x) + softDTW(y, y))
    """
    return (
        soft_dtw(x, y, gamma=gamma)
        - 0.5 * (soft_dtw(x, x, gamma=gamma) + soft_dtw(y, y, gamma=gamma))
    )


@dataclass(frozen=True)
class SoftDTWGIResult:
    P: np.ndarray
    loss_history: List[float]
    cost: float
    n_iter: int
    stopped_early: bool



def soft_dtw_gi(
    x: np.ndarray,
    y: np.ndarray,
    *,
    gamma: float = 1.0,
    max_iter: int = 50,
    lr: float = 1e-3,
    init_P: Optional[np.ndarray] = None,
    normalize: bool = True,
    early_stopping_patience: Optional[int] = 10,
    verbose: bool = False,
    step_verbose: int = 10,
) -> SoftDTWGIResult:
    """
    softDTW-GI implementation (Numpy, tslearn).

    Minimizes:
        softDTW(x, y @ P^T)
    or its normalized version, over P ∈ Stiefel.
    """

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D arrays (T, p).")

    Tx, px = x.shape
    Ty, py = y.shape

    # Initialization
    if init_P is None:
        P = np.eye(px, py)
    else:
        P = init_P.copy()

    P = _project_to_stiefel(P)

    best_loss = np.inf
    best_P = P.copy()
    best_iter = -1

    loss_history: List[float] = []
    stopped_early = False

    # Gradient descent loop
    for it in range(1, max_iter + 1):
        yP = _apply_linear_map(y, P)

        if normalize:
            loss = _normalized_softdtw(x, yP, gamma)
        else:
            loss = soft_dtw(x, yP, gamma=gamma)

        loss_history.append(loss)

        eps = 1e-6
        G = np.zeros_like(P)

        for i in range(px):
            for j in range(py):
                P_pert = P.copy()
                P_pert[i, j] += eps
                P_pert = _project_to_stiefel(P_pert)
                yP_pert = _apply_linear_map(y, P_pert)

                if normalize:
                    loss_pert = _normalized_softdtw(x, yP_pert, gamma)
                else:
                    loss_pert = soft_dtw(x, yP_pert, gamma=gamma)

                G[i, j] = (loss_pert - loss) / eps

        # gradient step + projection
        P = _project_to_stiefel(P - lr * G)

        # bookkeeping
        if loss < best_loss:
            best_loss = loss
            best_P = P.copy()
            best_iter = it

        if verbose and (it == 1 or it % step_verbose == 0):
            print(
                f"[softDTW-GI] iter={it:04d} | "
                f"loss={loss:.6f} | "
                f"best={best_loss:.6f} @ {best_iter:04d}"
            )

        if (
            early_stopping_patience is not None
            and best_iter >= 0
            and (it - best_iter) > early_stopping_patience
        ):
            stopped_early = True
            if verbose:
                print(f"[softDTW-GI] early stop @ iter={it:04d}")
            break

    final_cost = best_loss

    return SoftDTWGIResult(
        P=best_P,
        loss_history=loss_history,
        cost=float(final_cost),
        n_iter=len(loss_history),
        stopped_early=stopped_early,
    )

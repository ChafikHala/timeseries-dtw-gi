# dtw_gi/bcd.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np

from .dtw.backend import compute_dtw_path
from .dtw.alignment import path_to_W
from .stiefel import stiefel_procrustes_update, random_stiefel


@dataclass(frozen=True)
class BCDResult:
    P: np.ndarray                  # (px, py)
    path: List[Tuple[int, int]]    # DTW alignment on last iteration
    cost: float                    # DTW cost on last iteration (with transformed y)
    n_iter: int
    cost_history: List[float]



def dtw_gi(
    x: np.ndarray,
    y: np.ndarray,
    max_iter: int = 100,
    init_P: Optional[np.ndarray] = None,
    verbose: bool = False,
    backend: str = "tslearn"
) -> BCDResult:
    """
    BCD for DTW-GI when F is Stiefel (rigid linear map):
      - x: (Tx, px)
      - y: (Ty, py), assume px >= py
      - P: (px, py), and we compare x with y @ P^T in DTW step

    Iteration:
      1) path <- DTW(x, y P^T)
      2) W <- path_to_w(path)
      3) M <- x^T W y
      4) P <- stiefel_procrustes(M)
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D arrays: (T, p).")
    tx, px = x.shape
    ty, py = y.shape
    if px < py:
        raise ValueError(f"Require px >= py for Stiefel V_{py,px}. Got px={px}, py={py}.")

    if init_P is None:
        # Identity embedding: first py coordinates
        P = np.zeros((px, py), dtype=np.float64)
        P[:py, :py] = np.eye(py)
    else:
        P = np.array(init_P, dtype=np.float64, copy=True)
        if P.shape != (px, py):
            raise ValueError(f"init_P must have shape (px, py)=({px},{py}), got {P.shape}.")

    prev_cost = np.inf
    last_path: List[Tuple[int, int]] = []
    last_cost: float = np.inf
    prev_path = None
    cost_history = []

    for it in range(1, max_iter + 1):
        # DTW alignment with transformed y
        yP = y @ P.T  # (Ty, px)
        path, cost = compute_dtw_path(x, yP, backend)
        cost_history.append(cost)

        # build W and update P via SVD (Eq. 17 / Alg. 1)
        W = path_to_W(path, Tx=tx, Ty=ty)  # (Tx, Ty)
        res = stiefel_procrustes_update(x, y, W)
        P = res.P

        # convergence check
        if verbose:
            print(f"[BCD] iter={it:03d} cost={cost:.6f} Δ={prev_cost - cost:.6e}")

        if prev_path is not None and path == prev_path:
            break
        
        prev_path = path
        prev_cost = cost
        last_path, last_cost = path, float(cost)

    return BCDResult(P=P, path=last_path, cost=last_cost, n_iter=max_iter, cost_history=cost_history)




def dtw_gi_multistart(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_iter: int = 20,
    n_random_starts: int = 5,
    seed: int = 0,
    verbose: bool = False,
    backend: str = "tslearn"
) -> BCDResult:
    """
    Multi-start DTW-GI using BCD on the Stiefel manifold.

    Runs BCD from:
      - identity initialization
      - n_random_starts random Stiefel initializations

    Returns the solution with minimal final cost.
    """
    Tx, px = x.shape
    _, py = y.shape

    rng = np.random.default_rng(seed)

    candidates: list[BCDResult] = []

    # Identity initialization 
    P_id = np.eye(px, py)
    res_id = dtw_gi(
        x,
        y,
        max_iter=max_iter,
        init_P=P_id,
        verbose=False,
        backend=backend
    )
    candidates.append(res_id)

    if verbose:
        print(f"[Multi-start DTW-GI] identity cost = {res_id.cost:.6f}")

    #  Random Stiefel initializations 
    for k in range(n_random_starts):
        P0 = random_stiefel(px, py, rng)
        res = dtw_gi(
            x,
            y,
            max_iter=max_iter,
            init_P=P0,
            verbose=False,
            backend=backend,
        )
        candidates.append(res)

        if verbose:
            print(f"[Multi-start DTW-GI] random {k+1}/{n_random_starts} cost = {res.cost:.6f}")

    
    best = min(candidates, key=lambda r: r.cost)

    if verbose:
        print(
            f"[Multi-start DTW-GI] selected cost = {best.cost:.6f} "
            f"(n_iter={best.n_iter})"
        )

    return best
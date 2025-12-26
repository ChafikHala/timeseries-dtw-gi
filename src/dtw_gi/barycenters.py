from __future__ import annotations

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

from tslearn.metrics import soft_dtw


def project_to_stiefel(P: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(P, full_matrices=False)
    return U @ Vt


def apply_linear_map(y: np.ndarray, P: np.ndarray) -> np.ndarray:
    return y @ P.T


def normalized_softdtw(x, y, gamma):
    return (
        soft_dtw(x, y, gamma=gamma)
        - 0.5 * (soft_dtw(x, x, gamma=gamma) + soft_dtw(y, y, gamma=gamma))
    )



@dataclass(frozen=True)
class SoftDTWGIBarycenterResult:
    barycenter: np.ndarray
    Ps: List[np.ndarray]
    loss_history: List[float]
    n_iter: int


class SoftDTWGIBarycenter:
    """
    Paper implementation of Section 3.3:
    Barycenters under the (soft) DTW-GI geometry.
    """

    def __init__(
        self,
        T: int,
        p: int,
        gamma: float = 1.0,
        max_iter: int = 20,
        lr_b: float = 1e-2,
        early_stopping_patience: int | None = 5,
    ):
        self.T = T
        self.p = p
        self.gamma = gamma
        self.max_iter = max_iter
        self.lr_b = lr_b
        self.early_stopping_patience = early_stopping_patience

    def fit(
        self,
        dataset: List[np.ndarray],
        verbose: bool = True,
    ) -> SoftDTWGIBarycenterResult:

        n_series = len(dataset)
        px = dataset[0].shape[1]

        b = np.mean(dataset, axis=0)[: self.T, : self.p]
        Ps = [project_to_stiefel(np.eye(px, self.p)) for _ in dataset]

        best_loss = np.inf
        best_iter = -1
        loss_history = []

        for it in range(1, self.max_iter + 1):

            # Update each P_i
            for i, x_i in enumerate(dataset):
                P = Ps[i]

                eps = 1e-6
                G = np.zeros_like(P)

                for a in range(P.shape[0]):
                    for b_idx in range(P.shape[1]):
                        P_pert = P.copy()
                        P_pert[a, b_idx] += eps
                        P_pert = project_to_stiefel(P_pert)

                        loss_pert = normalized_softdtw(
                            x_i,
                            apply_linear_map(b, P_pert),
                            self.gamma,
                        )

                        loss = normalized_softdtw(
                            x_i,
                            apply_linear_map(b, P),
                            self.gamma,
                        )

                        G[a, b_idx] = (loss_pert - loss) / eps

                Ps[i] = project_to_stiefel(P - 1e-2 * G)

            # Update barycenter b
            grad_b = np.zeros_like(b)

            for i, x_i in enumerate(dataset):
                P = Ps[i]
                eps = 1e-6

                for t in range(self.T):
                    for d in range(self.p):
                        b_pert = b.copy()
                        b_pert[t, d] += eps

                        loss_pert = normalized_softdtw(
                            x_i,
                            apply_linear_map(b_pert, P),
                            self.gamma,
                        )
                        loss = normalized_softdtw(
                            x_i,
                            apply_linear_map(b, P),
                            self.gamma,
                        )

                        grad_b[t, d] += (loss_pert - loss) / eps

            grad_b /= n_series
            b -= self.lr_b * grad_b

            # bookkeeping
            total_loss = 0.0
            for i, x_i in enumerate(dataset):
                total_loss += normalized_softdtw(
                    x_i,
                    apply_linear_map(b, Ps[i]),
                    self.gamma,
                )
            total_loss /= n_series

            loss_history.append(total_loss)

            if verbose:
                print(
                    f"[Barycenter-GI] iter={it:03d} | "
                    f"loss={total_loss:.6f}"
                )

            if total_loss < best_loss:
                best_loss = total_loss
                best_iter = it

            elif (
                self.early_stopping_patience is not None
                and (it - best_iter) > self.early_stopping_patience
            ):
                if verbose:
                    print(f"[Barycenter-GI] early stop @ iter={it:03d}")
                break

        return SoftDTWGIBarycenterResult(
            barycenter=b,
            Ps=Ps,
            loss_history=loss_history,
            n_iter=len(loss_history),
        )

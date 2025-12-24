from __future__ import annotations

from typing import Iterable, List, Tuple

import torch


def path_to_W(path: Iterable[Tuple[int, int]], Tx: int, Ty: int, *, device=None) -> torch.Tensor:
    """
    Build alignment matrix Wπ ∈ {0,1}^{Tx×Ty} from a warping path.
    """
    W = torch.zeros((Tx, Ty), device=device, dtype=torch.float32)
    for i, j in path:
        W[i, j] = 1.0
    return W


def W_to_path(W: torch.Tensor) -> List[Tuple[int, int]]:
    """
    Extract (i,j) entries where W[i,j]=1. Assumes W is 0/1.
    """
    idx = torch.nonzero(W > 0.0, as_tuple=False)
    return [(int(i), int(j)) for i, j in idx]

# dtw_gi/gradient_descent.py
import torch
import torch.nn as nn
from typing import Optional, List
from dataclasses import dataclass

from .dtw.soft_dtw import soft_dtw
from .stiefel import apply_linear_map

@dataclass
class SoftDTWGIResult:
    P: torch.Tensor
    loss_history: List[float]
    final_cost: float

def soft_dtw_gi_stiefel(
    x: torch.Tensor,
    y: torch.Tensor,
    gamma: float = 0.1,
    max_iter: int = 100,
    lr: float = 1e-2,
    init_P: Optional[torch.Tensor] = None,
    verbose: bool = False
) -> SoftDTWGIResult:
    """
    Optimizes the softDTW-GI loss using gradient descent on the Stiefel manifold.
    f(y) = y @ P.T
    """
    Tx, px = x.shape
    Ty, py = y.shape
    device = x.device

    # 1. Initialize P
    if init_P is None:
        # Start with an identity-like mapping
        P = torch.zeros((px, py), device=device, requires_grad=True)
        with torch.no_grad():
            P[:py, :py] = torch.eye(py, device=device)
    else:
        P = init_P.clone().detach().requires_grad_(True)

    # 2. Use a simple SGD or Adam optimizer
    optimizer = torch.optim.Adam([P], lr=lr)
    
    loss_history = []

    for i in range(max_iter):
        optimizer.zero_grad()
        
        # Apply transformation: y_transformed = y @ P.T
        y_trans = apply_linear_map(y, P)
        
        # Compute soft-DTW loss 
        res = soft_dtw(x, y_trans, gamma=gamma)
        loss = res.cost
        
        # Backprop
        loss.backward()
        optimizer.step()

        # 3. Manifold Projection (Keep P in the Stiefel Manifold) [cite: 160]
        # P = U V^T where P = U S V^T
        with torch.no_grad():
            U, _, Vh = torch.linalg.svd(P, full_matrices=False)
            P.copy_(U @ Vh)

        loss_val = loss.item()
        loss_history.append(loss_val)
        
        if verbose and i % 10 == 0:
            print(f"[softDTW-GI] Iter {i:03d} | Loss: {loss_val:.6f}")

    return SoftDTWGIResult(P=P.detach(), loss_history=loss_history, final_cost=loss_history[-1])
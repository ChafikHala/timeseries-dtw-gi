import torch
import torch.nn as nn
from typing import List, Optional
from .dtw.soft_dtw import soft_dtw
from .stiefel import apply_linear_map

class SoftDTWGIBarycenter:
    def __init__(
        self, 
        T: int, 
        p: int, 
        gamma: float = 0.1, 
        max_iter: int = 50, 
        lr_b: float = 1e-2, 
        lr_P: float = 1e-2
    ):
        """
        Implementation of Section 3.3: Barycenters under the DTW-GI geometry.
        """
        self.T = T
        self.p = p
        self.gamma = gamma
        self.max_iter = max_iter
        self.lr_b = lr_b
        self.lr_P = lr_P

    def fit(self, dataset: List[torch.Tensor], verbose: bool = True):
        device = dataset[0].device
        n_series = len(dataset)
        
        # 1. Initialize the barycenter b randomly or as a mean
        # We use a simple normal distribution initialization
        b = torch.randn((self.T, self.p), device=device, requires_grad=True)
        
        # 2. Initialize a transformation Pi for each series in the dataset
        # Each Pi maps the series dimensionality to the barycenter dimensionality p
        Ps = []
        for x_i in dataset:
            pi_px = x_i.shape[1]
            # P_i: (pi_px, p) to map barycenter (T, p) -> (T, pi_px)
            P_i = torch.eye(pi_px, self.p, device=device).requires_grad_(True)
            Ps.append(P_i)
            
        optimizer = torch.optim.Adam([b] + Ps, lr=self.lr_b)

        for it in range(self.max_iter):
            optimizer.zero_grad()
            total_loss = 0
            
            for i, x_i in enumerate(dataset):
                # Map barycenter to x_i's space: b_trans = b @ P_i.T
                b_trans = apply_linear_map(b, Ps[i])
                
                # Compute soft-DTW between the original series and the transformed barycenter
                res = soft_dtw(x_i, b_trans, gamma=self.gamma)
                total_loss += res.cost
            
            total_loss /= n_series
            total_loss.backward()
            optimizer.step()

            # 3. Project Ps back to Stiefel Manifold to maintain rigid transformations
            with torch.no_grad():
                for P_i in Ps:
                    U, _, Vh = torch.linalg.svd(P_i, full_matrices=False)
                    P_i.copy_(U @ Vh)

            if verbose and it % 5 == 0:
                print(f"[Barycenter] Iter {it:03d} | Average Loss: {total_loss.item():.6f}")

        return b.detach(), Ps
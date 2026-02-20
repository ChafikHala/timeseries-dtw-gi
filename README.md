Implementation and experiments around **Dynamic Time Warping with Global Invariances (DTW-GI)** and its differentiable variant **softDTW-GI** for multivariate time series alignment, prototype estimation and forecasting.

This repository reproduces and studies the method introduced in  
*Time Series Alignment with Global Invariances* (TMLR, 2022): https://openreview.net/forum?id=JXCH5N4Ujy

---

### Overview

Comparing multivariate time series is challenging because two independent sources of variability coexist:

1. **Temporal variability** — signals may be stretched or compressed in time.
2. **Feature-space variability** — signals may differ by a global transformation (e.g. rotation, reflection, projection, sensor change).

Classical Dynamic Time Warping (DTW) handles only temporal distortions.  
DTW-GI addresses both by jointly estimating a temporal alignment and a global feature-space transformation.

Given two time series  
x ∈ ℝ^(T_x × p_x) and y ∈ ℝ^(T_y × p_y),

DTW-GI(x, y) =
min_{f ∈ F, π ∈ A(x,y)}  Σ_(i,j)∈π  || x_i − f(y_j) ||²

where:
- π is a valid DTW alignment path
- f is a global feature transformation

The soft version replaces the minimum over paths with a differentiable soft-minimum.

Instead of:

align time → compare features

DTW-GI performs:

align time AND align geometry simultaneously.

The learned transformation removes global geometric distortions (for example sensor rotations), while DTW accounts for temporal misalignment.

In practice, the transformation is chosen as an orthogonal matrix (Stiefel manifold) and is updated with a closed-form orthogonal Procrustes step.

---

### Optimization

#### DTW-GI (Block Coordinate Descent)

Alternates:
1. Compute DTW alignment with fixed transformation
2. Update transformation using SVD

This procedure is analogous to Iterative Closest Point but uses DTW correspondences instead of nearest neighbors.

#### softDTW-GI (Gradient-based)

The soft formulation is differentiable and optimized by backpropagating through the soft alignment recursion.

---

### What this repo studies

#### Alignment robustness
DTW-GI is invariant to global rotations, unlike classical DTW which is highly sensitive to them.

#### Elastic barycenters
Geometry-aware barycenters produce coherent prototypes under large geometric variability.

#### Spatiotemporal forecasting
Future trajectories are reconstructed using an attention-weighted combination of aligned training sequences, where weights depend on the DTW-GI similarity.


### Reference

@article{vayer2022dtwgi,
title={Time Series Alignment with Global Invariances},
author={Vayer, Titouan and Tavenard, Romain and Chapel, Laetitia and Flamary, Rémi and Courty, Nicolas and Soullard, Yann},
journal={Transactions on Machine Learning Research},
year={2022}
}
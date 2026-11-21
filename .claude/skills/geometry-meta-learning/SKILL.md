---
name: geometry-meta-learning
description: Train trajectory-curvature NeuralOpt with outcome-based multi-horizon objectives while preserving stochastic dynamics, SPD constraints, and checkpoint provenance.
---

Freeze architecture and task splits before expensive runs. Compare short history (0/1/4/8), no-attention curvature summary, matched MLP, and trajectory attention, with comparable parameter counts and training budgets. Use the current unrolled training, explicit JAX PRNG, gradient clipping and best-feasible multi-horizon regret.

Test an optional *soft secant-consistency auxiliary loss*, with a correctly defined direction (`B s≈y` for Hessian, `P y≈s` for inverse Hessian); estimate only on valid adequately conditioned pairs. Do not claim a low-rank SPD model matches all secants. Track objective/coordinate scaling, gradient variance, sparsity masks, preconditioner eigenvalue range, geometry reliability, and actual regret.

Do not let FP regularization or BAOAB constraints silently change with new features. Train stochastic mobility separately after the direct curvature feature experiment passes. Check that geometry updates change the right inverse/mass convention. Keep the existing fixed-diagonal baseline the winner unless new held-out evidence says otherwise.

Save model config, schema version, seed, train/val split hash, task generator config, training compute, artifact provenance. No test-set selection.

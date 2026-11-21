---
name: trajectory-curvature
description: Implement and test minimal masked secant-history features and optional learned curvature attention in the existing NeuralOpt JAX architecture without breaking older checkpoints.
---

Precondition: `/neuralopt-audit` and `/curvature-theory` completed, numerical design and experiment protocol agreed.

1. Add a fixed-length PyTree/ring buffer of **actually evaluated** `(u_t, g^u_t, f_t, validity)` records (and derived `s,y`) with bounded memory and fixed shapes. Ensure no gradient/objective duplication or stale-rejection state; JAX `scan`/`vmap` compatible.
2. Compute masked secant features: normalized displacement, gradient difference, `s^Ty`, `||s||`, `||y||`, alignment, span/conditioning and reliability. Protect zero norms, NaNs and scale transformations; retain negative-curvature diagnostics.
3. First test a **no-attention** history-statistic baseline using existing five-feature path plus opt-in model state. Only if it helps, add small temporal attention or cross-attention with 4/8 valid pairs. Keep coordinate permutation equivariance and chronological order explicit.
4. Version checkpoint schema and history mode; unchanged defaults must load all bundled legacy checkpoints and reproduce deterministic v0.5 inference within declared tolerance.
5. Verify exact quadratic `y=Hs` in normalized coordinates, gradient vs HVP, affine scaling, adversarial rejection, masked initialization, invalid/non-SPD curvature, `jit`/`vmap`/grad tests, memory/runtime.
6. Run targeted tests and collect first ablation evidence. Do not touch certification internals or add a classical optimizer fallback.

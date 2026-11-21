---
paths:
  - "src/neuralopt/features.py"
  - "src/neuralopt/model.py"
  - "src/neuralopt/optimizer.py"
  - "src/neuralopt/training.py"
  - "src/neuralopt/stochastic_geometry.py"
  - "src/neuralopt/hamiltonian.py"
  - "src/neuralopt/langevin.py"
  - "tests/**"
---
# Coordinate and curvature correctness

- Box normalization: `u = 2 (x-lo)/(hi-lo) - 1`. Write `J = diag((hi-lo)/2)`, so `g_u = J^T g_x`; Hessian `H_u = J^T H_x J`. Check gradients/HVPs and secant differences in the SAME frame.
- For two actually evaluated states, `s = u_next-u_previous`, `y = g_u(next)-g_u(previous)`; on a quadratic, `y = H_u s` (modulo numerical error). Check `y^T s`, span/conditioning, pair validity, objective scaling and noise.
- A secant pair with `y^T s <= 0` is not automatically erroneous: it may indicate negative curvature or noise. Do not feed it into a formula requiring SPD positive curvature without damping/skipping; preserve the diagnostic.
- Never enforce impossible exact Hessian identification from rank-deficient history. Test the rank of span(`s`) and report uncertainty/unobserved directions.
- Keep HMC mass and Langevin mobility conventions explicit; preserve correct momentum/noise covariance and training/inference coordinate consistency.
- Avoid dense `d x d` or `d x d x history` state in large-dimensional inference. Prefer small ring buffers and matrix-free products.
- All new learned features must have masks and strict checkpoint-schema/version handling. Retain old 5-feature model path unchanged by default.

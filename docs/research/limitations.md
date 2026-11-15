# Limitations

This v0.2 development prototype has no claim of universal superiority or production maturity. The bundled checkpoint remains the v0.1.0 supervised smoke model. The outcome-trained v0.2 smoke checkpoint failed its unseen-dimension test and is retained only as a negative result. Randomly initialized weights are labelled as such. Learned stopping is not used as a correctness claim.

The optimizer currently requires gradients and supports box constraints only. A JAX-differentiable objective uses compiled automatic differentiation; an explicit gradient keeps objective and gradient callbacks eager so NumPy or other host implementations remain supported. Residuals and nonlinear constraints are represented by the problem type but optimization rejects no constraint violations yet; they are forward-compatible metadata rather than implemented constrained optimization. Full history, curvature sketches, noise models, recurrent ablations, BBOB, MetaBox, ODE inversion, accelerator profiles, and faithful learned-optimizer baselines remain future experiments.

Certification supports an explicit expression language, not JAXPR. Its `nextafter` backend assumes correctly rounded host operations and libm; it is not IEEE 1788 conformant and cannot establish formal proof across arbitrary platforms. Replay checks retained bounds and geometric domain accounting but does not prove the implementation or hardware. Constraints, division, logarithms, arbitrary powers, interval Newton, Taylor models, and infeasibility proofs are unsupported and fail closed.


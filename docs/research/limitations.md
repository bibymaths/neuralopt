# Limitations

This 0.1.0 research prototype has no claim of universal superiority or production maturity. The bundled training curriculum is small and supervised; it does not establish transfer beyond its sampled quadratics. Randomly initialized weights are allowed for API experimentation but are labelled as such in benchmark records. Learned stopping is emitted for research but is not used as a correctness claim.

The optimizer currently requires gradients and supports box constraints only. Residuals and nonlinear constraints are represented by the problem type but optimization rejects no constraint violations yet; they are forward-compatible metadata rather than implemented constrained optimization. Full history, curvature sketches, noise models, recurrent ablations, BBOB, MetaBox, ODE inversion, accelerator profiles, and faithful learned-optimizer baselines remain future experiments.

Certification supports an explicit expression language, not JAXPR. Its `nextafter` backend assumes correctly rounded host operations and libm; it is not IEEE 1788 conformant and cannot establish formal proof across arbitrary platforms. The verifier replays retained leaves but does not prove the implementation or hardware. Constraints, division, logarithms, arbitrary powers, interval Newton, Taylor models, and infeasibility proofs are unsupported and fail closed.


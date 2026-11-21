---
name: curvature-theory
description: Derive trajectory-aware curvature representations for NeuralOpt from secant pairs, self-scaled Broyden, L-SR1, Optimus, LODO, and stochastic Hessian studies without introducing PINNs.
---

Read `docs/research/trajectory-curvature-literature.md` and verify primary sources when online. Translate each contribution into a testable action relevant to optimization over arbitrary JAX objectives (not PINN residual training).

For box-normalized states `u_t` and transformed gradients `g^u_t`, derive `s_t = u_{t+1}-u_t`, `y_t = g^u_{t+1}-g^u_t`, secant residual `||B s - y||`, inverse secant residual `||P y - s||`, and curvature scalar `s^T y`. Explain scaling/invariances, missing/noisy pairs, negative curvature and rank-deficient history.

Propose a *small* candidate set: (A) masked secant statistics, (B) low-rank online sketch as feature (not an inference solver), (C) temporal/set attention across valid pairs, (D) optional Hv-product probes counted against the budget, (E) curvature reliability gating. Rank by expressiveness, precision, compute and compatibility.

Require algebraic tests on diagonal and rotated quadratics and counterexamples on nonconvex/noisy landscapes. Do not claim learned SPD matrices can encode arbitrary indefinite Hessians or that one gradient reconstructs a full Hessian.

Deliver: concise derivation, paper-to-code matrix, failure modes, objective counterexample, and falsifiable gate. No new algorithm complexity without quantified need.

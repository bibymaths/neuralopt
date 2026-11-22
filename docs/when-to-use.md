# When to use NeuralOpt

**Short answer: to solve an optimization problem, use a classical solver. Use NeuralOpt
to study learned optimizers, or for its restricted certified mode.**

## Use SciPy or Optimistix when you want a good answer

- **SciPy `minimize(method="L-BFGS-B")`** for bounded, smooth problems on the CPU.
- **Optimistix (LBFGS, BFGS, …)** for JAX-native, `jit`/`vmap`-able solves.

In the v0.7 equal-work benchmark (32 rotated κ = 100 quadratics, d = 16, 100 units of
objective plus gradient work each), SciPy L-BFGS-B and Optimistix LBFGS reached a
regret of 1e-6 on 32 of 32 tasks. NeuralOpt's learned geometry reached 1e-3 on 10 of 32
and 1e-6 on none ([engine-speed report](research/engine-speed-v0.7.md)). Across v0.4–v0.7
the learned geometries did not beat validation-tuned fixed-diagonal or
Barzilai–Borwein steps ([release notes](release-notes/index.md)).

## Use NeuralOpt when you want to

- **Research learned optimizers** with honest accounting: a shared evaluation budget,
  masked histories built only from evaluated states, versioned checkpoint schemas,
  frozen pre-registrations and paired comparisons against fixed and classical
  references.
- **Run a learned update rule fast**: the [compiled engine](guide/compiled-engine.md)
  runs the loop as one XLA program and `vmap`s over task families.
- **Get a replayable global certificate** for a small expression in the supported
  subset (`+`, `-`, `*`, square, `sin`, `exp`) on a box, with
  [certified mode](guide/certified-mode.md). For general global optimization, use a
  dedicated deterministic global solver.

## Do not use NeuralOpt for

- Production optimization, or any claim that a learned step is better than L-BFGS.
- Certifying arbitrary JAX programs: certified mode is restricted to its expression
  language, and neural or stochastic outputs never certify optimality.
- Constrained problems beyond box bounds.

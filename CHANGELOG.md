# Changelog

## 0.7.0 - In development

- Track A (speed only, opt-in): `preconditioned_engine="compiled"` runs the v0.6 loop as
  one `lax.while_loop` (float64 parity with the Python loop to 1e-10), with a vmapped
  batch API; parametric task families and a compiled, batched, NaN-guarded trainer
  (`train_batched`, skipped updates counted). Engine and training benchmarks in
  `results/engine-benchmark-v0.7.0.json`; Optimistix added to the `benchmark` extra only.
- Track B (opt-in, research): Gram-space history model (`model_kind="gram"`, schema
  gram-v2), opt-in step-acceptance rules (`loop_rule`), gates 1-2, loop-rule ablation,
  mask diagnostic, a frozen pre-registration (sha256 recorded) and its training and
  evaluation infrastructure. Pre-registered result (negative): Stage 1 passed, but
  gram-full did not beat the untrained BB step on the curvature domain (H1 fails), the
  reliability guard trips on Rastrigin and Ackley, and the model is far from the
  oracle-BFGS ceiling. BB stays the bar.

## 0.6.0 - In development

- Add opt-in `history-v1` trajectory features (`neuralopt.trajectory`): a bounded buffer
  of actually evaluated normalized `(u, g_u, f)` records, masked secant pairs and twelve
  scale-invariant curvature/reliability features; versioned checkpoint schema that fails
  closed, with v0.1--v0.5 checkpoints and outputs pinned by a golden regression test.
- Add `candidate_method="preconditioned"`, a shared loop charging objective and gradient
  calls to one budget, and `train(objective="history")`, its differentiable mirror.
- Pre-registered multi-seed study with a negative main result: an untrained
  Barzilai--Borwein step beats every learned arm by about one decade, and attention adds
  nothing over an MLP (H2). History plus a rejection signal helps over a matched control
  (H1); under a 3.5x anisotropy cap, learned geometry does not beat the tuned fixed step
  under the reliability rule (H3). Fixed geometry stays the default.

## 0.5.0 - In development

- Add scale-aware, bounded diagonal-plus-low-rank geometry while preserving
  legacy checkpoint behavior.
- Add diagonal, low-rank, full, FP, and training-budget ablations with disjoint
  validation tuning and a 30-instance held-out comparison.
- Reproduce v0.4 exactly and reconcile its pre-review and corrected FP results.
- Retain the negative result that learned geometry did not beat tuned fixed
  geometry overall, with family-level failure analysis.

## 0.4.0 - In development

- Train transformer diagonal-plus-low-rank mobility and mass heads through
  differentiable overdamped and BAOAB trajectories.
- Add multi-horizon stochastic regret, stability, fixed compute-cost, robust
  weak-generator, and optional Gaussian free-energy-surrogate losses.
- Add identity, fixed-diagonal, untrained, regret-trained, and
  regret-plus-Fokker--Planck ablations with held-out paired confidence intervals.
- Preserve negative results: learned BAOAB geometry does not beat the fixed
  diagonal baseline, and Fokker--Planck transfer to BAOAB is inconclusive.

## 0.3.0 - In development

- Add opt-in overdamped Euler--Maruyama and underdamped BAOAB candidate
  trajectories with identity or learned diagonal-plus-low-rank geometry.
- Add exact small-dimensional Itô mobility divergence, weak Fokker--Planck
  diagnostics, and an analytic Gaussian free-energy reference.
- Add a periodic Gaussian reference and mesh-normalized Fourier reconstruction
  experiment across four resolutions.
- Add an optional Diffrax reference extra and retain negative learned-geometry
  smoke results with explicit objective, gradient, integration, and timing costs.
- Record the blocked distributional-training, learned function-space transfer,
  and certification-injection gates without changing certification lower bounds.

## 0.2.0 - In development

- Add evaluation-neutral trajectory diagnostics and correct stale state after rejection.
- Add normalized coordinate transforms, differentiable multi-horizon regret unrolling,
  finite meta-gradient tests, and a matched-input MLP ablation.
- Add deterministic certification scheduling controls and strengthen independent replay
  with pruned-leaf domain coverage and non-overlap checks.
- Preserve a negative outcome-training smoke checkpoint without promoting it.
- Add opt-in native JAX Hamiltonian proposals with transformed box constraints,
  fixed or learned mass geometry, separate exploration/incumbent state, explicit
  cost accounting, correctness tests, and matched-budget smoke evidence.

## 0.1.0 - Unreleased

- Add direct transformer optimization policy, deterministic synthetic training, portable checkpoints, bounded problem API, and restricted interval certificates.
- Add model-independent certificate replay, tests, research audit, honest smoke benchmark, and packaging metadata.


# Changelog

All notable changes to NeuralOpt. Each version has curated [release notes](https://bibymaths.github.io/neuralopt/latest/release-notes/);
dates are UTC merge dates. Versions before 0.7.0 were tagged retroactively and are not on PyPI.

## [Unreleased]

- Nothing yet.

## [0.7.0] - 2026-11-22

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
  Exploratory post-mortem (training and validation tasks only): wider models imitate
  the oracle better but none performs well in the loop, and A/C gradients reach the
  heads without a saddle, so the cause is undetermined (leaning optimization signal).
  By plan gate G7 the line is not expanded.
- **Breaking:** `minimize(..., preconditioned_engine=...)` now defaults to `"compiled"`;
  `candidate_method="preconditioned"` with a host/NumPy `gradient=` raises a `ValueError`
  naming `preconditioned_engine="python"`. The v0.6 reproduction script pins
  `"python"` (its sha256 changed; frozen v0.6 results unchanged).
- **Breaking:** `neuralopt.__version__` comes from the package metadata;
  `pyproject.toml` is the single version source (`.bumpversion.toml` removed).
- Next direction (2026-11-22): amortized optimization over problem families; learned
  guidance for certified search is deferred, not cancelled.
- Release engineering: first PyPI release (trusted publishing), tag-driven GitHub
  releases, versioned docs (Zensical with its mike fork), curated release notes for
  every version, git-cliff with conventional PR titles, release-metadata checks on every
  PR, updated `CITATION.cff` and `.zenodo.json`, Citing page and logo.

PRs #7, #8 and the release-engineering PR · [release notes](https://bibymaths.github.io/neuralopt/latest/release-notes/v0.7.0/)

## [0.6.0] - 2026-11-21

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

PR #6 · [release notes](https://bibymaths.github.io/neuralopt/latest/release-notes/v0.6.0/)

## [0.5.0] - 2026-11-20

- Add scale-aware, bounded diagonal-plus-low-rank geometry while preserving
  legacy checkpoint behavior.
- Add diagonal, low-rank, full, FP, and training-budget ablations with disjoint
  validation tuning and a 30-instance held-out comparison.
- Reproduce v0.4 exactly and reconcile its pre-review and corrected FP results.
- Retain the negative result that learned geometry did not beat tuned fixed
  geometry overall, with family-level failure analysis.

PR #5 · [release notes](https://bibymaths.github.io/neuralopt/latest/release-notes/v0.5.0/)

## [0.4.0] - 2026-11-15

- Train transformer diagonal-plus-low-rank mobility and mass heads through
  differentiable overdamped and BAOAB trajectories.
- Add multi-horizon stochastic regret, stability, fixed compute-cost, robust
  weak-generator, and optional Gaussian free-energy-surrogate losses.
- Add identity, fixed-diagonal, untrained, regret-trained, and
  regret-plus-Fokker--Planck ablations with held-out paired confidence intervals.
- Preserve negative results: learned BAOAB geometry does not beat the fixed
  diagonal baseline, and Fokker--Planck transfer to BAOAB is inconclusive.

PR #4 · [release notes](https://bibymaths.github.io/neuralopt/latest/release-notes/v0.4.0/)

## [0.3.0] - 2026-11-15

- Add opt-in overdamped Euler--Maruyama and underdamped BAOAB candidate
  trajectories with identity or learned diagonal-plus-low-rank geometry.
- Add exact small-dimensional Itô mobility divergence, weak Fokker--Planck
  diagnostics, and an analytic Gaussian free-energy reference.
- Add a periodic Gaussian reference and mesh-normalized Fourier reconstruction
  experiment across four resolutions.
- Add an optional Diffrax reference extra and retain negative learned-geometry
  smoke results with explicit objective, gradient, integration, and timing costs.
- Record the gate status: distributional training and learned function-space transfer
  partial, certification injection blocked; certification lower bounds unchanged.

PR #3 · [release notes](https://bibymaths.github.io/neuralopt/latest/release-notes/v0.3.0/)

## [0.2.0] - 2026-11-15

- Add evaluation-neutral trajectory diagnostics and correct stale state after rejection.
- Add normalized coordinate transforms, differentiable multi-horizon regret unrolling,
  finite meta-gradient tests, and a matched-input MLP ablation.
- Add deterministic certification scheduling controls and strengthen independent replay
  with pruned-leaf domain coverage and non-overlap checks.
- Preserve a negative outcome-training smoke checkpoint without promoting it.
- Add opt-in native JAX Hamiltonian proposals with transformed box constraints,
  fixed or learned mass geometry, separate exploration/incumbent state, explicit
  cost accounting, correctness tests, and matched-budget smoke evidence.

PR #2 · [release notes](https://bibymaths.github.io/neuralopt/latest/release-notes/v0.2.0/)

## [0.1.0] - 2026-11-14

- Add direct transformer optimization policy, deterministic synthetic training, portable checkpoints, bounded problem API, and restricted interval certificates.
- Add model-independent certificate replay, tests, research audit, honest smoke benchmark, and packaging metadata.

PR #1 · [release notes](https://bibymaths.github.io/neuralopt/latest/release-notes/v0.1.0/)

[Unreleased]: https://github.com/bibymaths/neuralopt/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/bibymaths/neuralopt/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/bibymaths/neuralopt/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/bibymaths/neuralopt/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/bibymaths/neuralopt/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/bibymaths/neuralopt/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/bibymaths/neuralopt/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/bibymaths/neuralopt/releases/tag/v0.1.0

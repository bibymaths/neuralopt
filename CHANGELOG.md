# Changelog

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


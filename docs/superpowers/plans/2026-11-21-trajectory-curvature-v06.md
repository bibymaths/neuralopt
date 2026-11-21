# Trajectory-curvature v0.6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Opt-in masked secant-history features, a shared preconditioned evaluation loop, history meta-training, and a pre-registered matched-budget ablation against tuned fixed geometry.

**Architecture:** New `trajectory.py` (history PyTree + features); `ModelConfig` gains a versioned feature schema; `minimize` gains `candidate_method="preconditioned"`; `training.py` gains `history_regret_loss` and checkpoint schema; one experiment script produces versioned artifacts and results.

**Tech Stack:** JAX 0.12 CPU, Optax, NumPy, SciPy (harness only), pytest, uv.

**Spec:** `docs/superpowers/specs/2026-11-21-trajectory-curvature-v06-design.md`

## Global Constraints

- Legacy defaults (`feature_set="legacy"`, existing candidate methods) produce outputs identical to the pre-change golden snapshot (rtol 1e-6).
- No new keys in legacy checkpoint files; unknown schema raises `ValueError`.
- No classical optimizer in `src/neuralopt`; BB and L-BFGS live only in `scripts/`.
- History entries only from states with objective and gradient evaluated at that exact point.
- Never write `results/*v0.5*` or existing `artifacts/*.npz`.
- No commits/pushes unless the user asks.

## Review Focus

- Problem with all candidates rejected (flat/plateau): loop must terminate within budget, history stays one entry, no NaN features.
- Candidate clipped to the box: secant uses the clipped (evaluated) point.
- Explicit NumPy `gradient` callables: counts must match actual calls.
- Objective returning NaN/inf at a candidate: rejected, never pushed.
- Dimension 1 problems: features and effective-rank finite.

---

### Task 1: Legacy golden snapshot and regression test

**Files:** Create `tests/legacy_cases.py`, `tests/data/legacy-v0.5-golden.json`, `tests/test_legacy_regression.py`

- [ ] Write `legacy_outputs() -> dict` covering: default `minimize` (demo checkpoint) direct 4-d quadratic budget 30; v0.5 `overdamped_full` langevin learned; v0.5 `underdamped_full` underdamped learned; langevin fixed metric (constant-metric stochastic); hamiltonian identity; `apply_model` outputs for every `artifacts/*.npz` and bundled checkpoint on fixed 3-d features; loaded configs.
- [ ] Generate golden JSON with current (unmodified) code.
- [ ] Test compares `legacy_outputs()` to golden (rtol 1e-6); run → PASS before any source change.

### Task 2: History buffer and secant features (information gate)

**Files:** Create `src/neuralopt/trajectory.py`, `tests/test_trajectory.py`

**Produces:** `TrajectoryHistory` NamedTuple; `empty_history(pairs:int, dimension:int, dtype) -> TrajectoryHistory`; `push_history(h, u, g_u, f) -> TrajectoryHistory`; `record_rejection(h) -> TrajectoryHistory`; `secant_pairs(h) -> (S, Y, mask)`; `history_features(h, g_u, pairs:int) -> (d, 12)`; `HISTORY_FEATURES = 12`.

- [ ] Tests (fail first): exact `y = H_u s` on rotated SPD quadratic with unequal box widths (u-frame Jacobian); diagonal quadratic ⇒ `a_i/ρ` equals `h_i/ρ`; non-identifiability (two Hessians equal on span(S) ⇒ identical features); objective-scale and box-scale invariance; coordinate-permutation equivariance; negative curvature fraction on indefinite quadratic; repeated point and roundoff-level pair masked; empty history ⇒ masked features zero, flags correct; rejection increments counter without pushing; NaN pair masked; `jit`, `vmap` over histories, `grad` finite; d = 1 finite.
- [ ] Implement; tests PASS.

### Task 3: Versioned model schema and checkpoints

**Files:** Modify `src/neuralopt/model.py`, `src/neuralopt/training.py` (save/load), `src/neuralopt/optimizer.py` (`_validate_policy_params`); test `tests/test_trajectory.py`

**Produces:** `ModelConfig.feature_set`, `ModelConfig.history_length`, `ModelConfig.input_features`; `HISTORY_SCHEMA = "neuralopt-policy/v0.6-history-v1"`.

- [ ] Tests (fail first): history config init shapes (embed 17×w, MLP hidden 19×w); legacy config rejects `history_length>0`; history roundtrip preserves schema fields; legacy save writes no `schema` key; unknown schema raises; legacy params with history config raise on validation.
- [ ] Implement; tests + golden regression PASS.

### Task 4: Shared preconditioned loop

**Files:** Modify `src/neuralopt/optimizer.py`; test `tests/test_trajectory.py`

**Consumes:** Task 2/3 APIs. **Produces:** `minimize(..., candidate_method="preconditioned", preconditioned_step: float | None = None)`; diagnostics keys `objective_evaluations`, `gradient_evaluations`, `hvp_evaluations`, `accepted_steps`, `history` (dict of arrays when `record_diagnostics`).

- [ ] Tests (fail first): counting wrapper (NumPy objective + gradient) — counted calls equal reported counts and `obj+grad ≤ budget`; every history entry satisfies `f == f(x)`, `g == grad(x)` at denormalized `u`; all-reject objective terminates with one entry and growing rejections; clipped candidate stays in bounds and is the stored point; NaN candidate rejected; fixed geometry requires `preconditioned_step`; history config with stochastic method raises; learned history model runs d = 2/4/8/10; legacy golden still PASS.
- [ ] Implement as a separate private function called from `minimize`.

### Task 5: History meta-training

**Files:** Modify `src/neuralopt/training.py`; test `tests/test_trajectory.py`

**Produces:** `HistoryUnrollConfig(steps=24, horizons=(6,12,24), candidate_weight=0.1)`; `history_regret_loss(params, problem, key, config, unroll, model_kind) -> scalar`; `train(..., objective="history", history_unroll=HistoryUnrollConfig())`.

- [ ] Tests (fail first): finite nonzero meta-gradient for transformer and MLP, history-v1 K=4 and legacy; loss `jit`s; `train` with 3 steps runs; acceptance in the scan matches the inference rule on a toy case.
- [ ] Implement; full `pytest`, ruff, pyright PASS.

### Task 6: Experiment harness, timing probe, runs

**Files:** Create `scripts/run_trajectory_curvature_v0_6.py`

- [ ] Task generator, splits with hashes, fixed-step validation tuning, BB and L-BFGS-B references (harness only), parallel training per (arm, seed) via processes, evaluation at budgets 50/100/200, two-level bootstrap, JSON output.
- [ ] Timing probe on training tasks only (one arm, 20 updates); keep 400 updates unless an arm projects > 15 min.
- [ ] Run full protocol; write artifacts and results.

### Task 7: Report, review, quality gate

**Files:** Create `docs/research/trajectory-curvature-v0.6.md`; modify `CHANGELOG.md`, `artifacts/README.md`, `docs/plans/neuralopt-v06-trajectory-curvature.md` (status lines)

- [ ] Write report strictly from results JSON (H1/H2/H3 outcomes, per-family, costs, deviations).
- [ ] Dispatch `curvature-reviewer` and `experiment-auditor`; address findings.
- [ ] `uv run pytest -q`, `ruff check .`, `pyright`, `zensical build`, `uv build`, `twine check`.

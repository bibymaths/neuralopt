# v0.6 trajectory-aware curvature: pre-registered result

## Result

**This is a negative result.** The main finding has two parts:

- **The best learned arm loses to an untrained secant step.** A Barzilai–Borwein (BB)
  step-length rule uses the same loop, history and 100-unit budget, with no training.
  It beats every learned arm by about one decade of normalized regret. For the 8-pair
  history transformer on the curvature domain, tf-h8 − BB = +0.92 [+0.59, +1.25]
  (positive means BB is better).
- **Attention adds nothing over an MLP (H2).**

Learned geometry does not beat the validation-tuned fixed step under the
pre-registered rule. Nothing here supports a claim that learned curvature geometry is
competitive with quasi-Newton methods: L-BFGS-B leads by about six decades. The v0.5
negative result stands, and fixed geometry remains the default.

The study was pre-registered in
`docs/superpowers/specs/2026-11-21-trajectory-curvature-v06-design.md`, written before
any training; its sha256 is recorded in the results file. Hypotheses are tested on
the curvature domain (diagonal κ=1000, rotated κ=100 and rotated κ=1000 at unseen
dimensions 7/10/16) at budget 100:

- **H1: history plus a rejection signal helps. Supported.** The 8-pair history
  transformer beat its matched no-history control by 0.54 decades.
  - The no-history control drops both the secant features and the rejection counter.
  - So this shows that the history *bundle* helps. It does not isolate the value of
    secant pairs.
- **H2 (attention value): not supported.** Cross-coordinate attention added nothing
  measurable over a no-attention MLP fed the same history features.
- **H3 (beats tuned fixed geometry): not supported, under the 3.5× anisotropy cap.**
  - On the curvature domain the learned geometry beat the fixed scalar step by 0.22
    decades.
  - It failed the pre-registered reliability guard because it regressed on Rastrigin
    and Ackley.
  - The pre-registered parameterization caps off-axis anisotropy at about 3.5×
    (limitation 1). Problems needing 100–1000× therefore did not test what learned
    geometry could do without that cap.

v0.6 adds one positive, narrower finding: a history of actually evaluated states,
together with a rejection signal, carries information a small learned policy can use.
Even with it, the policy does not reach an untrained secant step.

## What was built (opt-in; legacy defaults unchanged)

- **`neuralopt.trajectory`:** a fixed-shape shift buffer of
  `(u, g_u, f)` records. Each record is evaluated at the same point, in normalized box
  coordinates.
  - Only accepted states are stored. Rejected candidates only increment a counter.
  - From the buffer it derives masked secant pairs `s = Δu`, `y = Δg_u`, and twelve
    objective-scale-invariant, coordinate-permutation-equivariant `history-v1` features.
- **`ModelConfig(feature_set="history-v1", history_length=K)`:** a versioned schema.
  - Checkpoints carry `schema = "neuralopt-policy/v0.6-history-v1"`.
  - Unknown, stripped or inconsistent schemas fail closed.
  - Legacy files are written byte-compatibly, and all v0.1–v0.5 assets reproduce a
    golden snapshot taken before any change (`tests/test_legacy_regression.py`).
- **`minimize(..., candidate_method="preconditioned")`:** one loop shared by every arm.
  - Update: `u' = clip(u − σ L P ĝ, −1, 1)`.
  - Hard acceptance; `σ` halves after a rejection.
  - Objective **plus** gradient calls are charged to the budget, and a gradient is taken
    once per accepted state.
  - HVP count is 0.
- **`train(..., objective="history")`:** a `lax.scan` mirror of that loop. Its
  trajectory and stored history are tested equal to inference.
- HMC, Langevin, BAOAB, Fokker–Planck, function-space and certification code is
  unmodified. Stochastic methods reject history configurations (G5 is gated).

### Identifiability (documented and tested)

On a quadratic, `y = H_u s` with `H_u = J H_x J`, `J = diag((hi−lo)/2)`
(`test_secant_is_exact_hessian_action_in_normalized_frame`). K pairs identify `H_u`
only on `span(S)`, with rank at most `min(K, d)`.

`test_history_cannot_distinguish_hessians_agreeing_on_observed_span` builds two
Hessians that differ by 50× outside the observed span. Starting from the same initial
record, they yield identical features. So no feature here implies full-Hessian
recovery.

Negative `sᵀy` is reported as a diagnostic (`test_negative_curvature_is_reported_not_hidden`).
It is never inverted. The geometry `P = diag + UUᵀ` stays SPD.

## Protocol (as pre-registered)

- **Task splits:**
  - Train: 21 tasks, dimensions 2/4/8, seeds 1000+.
  - Validation: 14 tasks, dimensions 3/6, seeds 2000+.
  - Final test: 84 tasks, unseen dimensions 7/10/16, seeds 3000+. Test tasks were
    built only after training and validation tuning.
- **Held-out families:** rotated κ=1000, Rosenbrock and Ackley.
- **Seeds and comparator:**
  - Three training seeds per arm; 400 Adam updates each; the final weights are used.
  - The fixed comparator is a scalar normalized-gradient step `L` with `P = I`. `L`
    was selected on validation only (0.03 from {0.01, 0.03, 0.1, 0.3, 1}).
- **Metric:** `D = log10(r_cand + 1e-10) − log10(r_ref + 1e-10)` with
  `r = (f − f*)/|f₀ − f*|`. Negative values favour the candidate.
- **Confidence intervals:** 95% percentile intervals from a two-level bootstrap with
  10,000 draws.
  - Each draw resamples **both** the three training seeds and the test problems, with
    replacement and independently (crossed, not nested), then averages the paired
    differences.
  - When both arms are untrained references (fixed, BB, L-BFGS-B), there is no seed
    level, so only problems are resampled.
  - With only 3 training seeds, the seed level of the bootstrap captures training
    variation only coarsely.
- **L-BFGS-B budget:** every SciPy objective call, line-search trials included, is
  charged one objective plus one gradient unit against the same budget.
  - Calls are hard-capped at budget/2, so it never exceeds 100 units at budget 100.
  - It averaged 82.7 units, because L-BFGS-B sometimes stops early, after as few as 3
    calls.

## Primary hypotheses (budget 100 units)

| Hypothesis | Contrast (curvature domain, budget 100) | Mean D (95% CI) | Wins | Pre-registered outcome |
|---|---|---:|---:|---|
| H1 (history + rejection signal) | tf-h8 − tf-h0 | -0.54 [-0.80, -0.32] | 103/108 | supported |
| H2 | tf-h8 − mlp-h8 | -0.08 [-0.20, +0.06] | 63/108 | not supported |
| H3 (under 3.5× cap) | tf-h8 − fixed | -0.22 [-0.43, -0.03] | 84/108 | not supported |

H3's curvature-domain interval excludes zero, but the pre-registered rule also
requires that no test family regress by more than 0.5 decades with an interval above
zero:

| Family | tf-h8 − fixed (95% CI) |
|---|---:|
| isotropic | -0.31 [-0.79, +0.10] |
| diagonal-1000 | -0.28 [-0.77, +0.16] |
| rotated-100 | -0.17 [-0.48, +0.06] |
| rastrigin | **+2.39 [+0.25, +4.78]** |
| rotated-1000 | -0.21 [-0.37, -0.07] |
| rosenbrock | -0.01 (not interpretable; see limitations) |
| ackley | **+0.82 [+0.20, +1.48]** |

On Rastrigin, 28% of tf-h8 runs end outside the global basin (log10 regret > −3). The
fixed step never does: its smaller steps from the centred start stay in the global
basin.

### Mean log10 normalized regret per family (budget 100)

| Arm | isotropic | diagonal-1000 | rotated-100 | rastrigin | rotated-1000 | rosenbrock | ackley |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed | -9.48 | -2.46 | -2.55 | -10.00 | -2.21 | -0.08 | -6.09 |
| tf-legacy | -7.90 | -2.16 | -2.11 | -9.80 | -2.03 | -0.05 | -4.96 |
| tf-h0 | -8.61 | -2.15 | -2.12 | -9.68 | -2.00 | -0.05 | -4.85 |
| tf-h4 | -9.82 | -2.67 | -2.66 | -7.99 | -2.29 | -0.09 | -5.21 |
| tf-h8 | -9.80 | -2.74 | -2.72 | -7.61 | -2.42 | -0.09 | -5.27 |
| mlp-h0 | -9.83 | -2.29 | -2.22 | -9.16 | -2.09 | -0.04 | -5.44 |
| mlp-h4 | -9.94 | -2.67 | -2.77 | -6.64 | -2.36 | -0.08 | -5.82 |
| mlp-h8 | -9.97 | -2.51 | -2.75 | -7.08 | -2.38 | -0.07 | -5.87 |
| bb (reference) | -10.00 | -2.96 | -4.75 | -9.33 | -2.94 | -0.19 | -6.44 |
| lbfgs (reference) | -10.00 | -8.08 | -9.96 | -6.27 | -7.41 | -2.43 | -6.05 |
| v05-frozen (context) | -8.89 | -2.12 | -2.26 | -10.00 | -2.14 | -0.06 | -5.43 |

### Secondary contrasts (exploratory; curvature domain)

| Candidate | Reference | Budget 50 | Budget 100 | Budget 200 |
|---|---|---:|---:|---:|
| tf-h0 | fixed | +0.11 [-0.01, +0.24] | +0.32 [+0.20, +0.45] | +0.36 [+0.21, +0.54] |
| tf-h8 | fixed | -0.26 [-0.44, -0.09] | -0.22 [-0.43, -0.03] | -0.31 [-0.57, -0.06] |
| mlp-h4 | fixed | -0.23 [-0.33, -0.12] | -0.19 [-0.32, -0.06] | -0.40 [-0.57, -0.22] |
| tf-h8 | bb | +0.24 [+0.04, +0.42] | +0.92 [+0.59, +1.25] | +2.31 [+1.63, +3.01] |
| tf-h8 | lbfgs | +3.46 [+2.80, +4.18] | +5.85 [+5.18, +6.50] | +6.65 [+6.35, +6.94] |
| bb | fixed | -0.50 [-0.61, -0.38] | -1.14 [-1.43, -0.86] | -2.62 [-3.33, -1.95] |
| v05-frozen | fixed | +0.04 [-0.05, +0.13] | +0.23 [+0.14, +0.35] | +0.26 [+0.14, +0.41] |

| Contrast (curvature, budget 100) | Mean D (95% CI) |
|---|---:|
| tf-h4 − tf-h0 | -0.45 [-0.67, -0.29] |
| mlp-h8 − mlp-h0 | -0.35 [-0.53, -0.17] |
| mlp-h4 − mlp-h0 | -0.40 [-0.53, -0.28] |
| tf-h8 − tf-h4 | -0.09 [-0.20, +0.02] |
| tf-h0 − tf-legacy | +0.01 [-0.05, +0.06] |
| tf-h4 − mlp-h4 | +0.06 [-0.13, +0.23] |

The history gain is consistent across both architectures and both K, and K=8 is not
clearly better than K=4. The history-free arms (tf-legacy, tf-h0, mlp-h0) and the
frozen v0.5 checkpoint are all worse than the fixed step. The learned arms' advantage
over fixed is smaller than BB's. That makes adaptive step length a plausible
explanation for most of it, rather than learned anisotropic geometry (see limitation 1).

### Costs (budget 100; all arms ≤ 100 units, no failures, no HVPs)

| Arm | Obj | Grad | Success 1e-3 | First call s (median/max) | Warm s |
|---|---:|---:|---:|---:|---:|
| fixed | 66.0 | 33.2 | 0.46 | 0.16/1.37 | 0.155 |
| tf-h0 | 80.0 | 19.1 | 0.44 | 0.21/2.58 | 0.245 |
| tf-h8 | 76.8 | 22.4 | 0.50 | 0.22/3.24 | 0.257 |
| mlp-h8 | 78.5 | 20.6 | 0.46 | 0.22/3.16 | 0.252 |
| bb | 69.0 | 24.9 | 0.68 | 0.30/3.01 | 0.448 |
| lbfgs | 41.3 | 41.3 | 0.85 | 0.09/0.97 | 0.099 |

These timings are descriptive only:
- 25 evaluation processes ran concurrently.
- "First call" is the first run per task, so it includes objective compilation. The
  maximum includes model compilation in a fresh process.
- `bb` runs through an un-jitted harness loop. L-BFGS calls jitted value-and-gradient
  closures from SciPy.

Training cost:
- 71–108 s per arm and seed in a single CPU process; 116 s wall time for all 21 runs in
  parallel.
- Parameter counts: transformer 1,488; MLP 1,392; tf-legacy 1,296.
- Training uses second-order autodiff, because the meta-gradient flows through `∇f`.
  That cost is not included in the inference counts above.

## Limitations and threats to validity

1. **Geometry capacity.** The pre-registered parameterization inherited from v0.5
   (`geometry_bound=10`, `lowrank_scale=0.5`) caps the low-rank term at
   `λmax(UUᵀ) < 0.25`. Off-axis anisotropy is therefore at most about 3.5×, while
   rotated κ=100–1000 needs 100–1000×. History can mainly steer the diagonal and the
   step length. The H3 curvature-domain test is therefore weaker than the research
   question implies.
2. **Train/test mismatch.** Training used a 24-candidate unroll (budget feature over 50
   units), random starts and a 1e-6 regret floor. Evaluation used budgets 50/100/200,
   centred starts and a 1e-10 floor.
   - The 50/100 budget mismatch is expected to bias H3 and the BB comparison against
     the learned arms. The fixed step was tuned at budget 100 and BB has no budget
     input, while learned arms read a remaining-budget feature stretched to twice the
     horizon they were trained on.
   - It does not bias H1 or H2, which compare learned arms that share the same
     mismatch.
   - The size of the bias was not measured.
3. **float32 floor.** 701 of 6,300 runs reach exactly the optimum in float32, so the
   1e-10 floor shapes means, especially for isotropic, Rastrigin and Ackley.
   - A **post-hoc** recomputation with a 1e-6 floor (not pre-registered;
     `--tables`) leaves the verdict unchanged.
   - Curvature domain: −0.22 [−0.43, −0.03]. Rastrigin: +1.27 [+0.14, +2.55]. Ackley:
     +0.58 [+0.14, +1.10].
4. **Rosenbrock replicates.** `analytic_task` ignores the seed for Rosenbrock and starts
   at the box centre. Its four replicates per dimension differ only in scale, and
   replicate 3 duplicates replicate 0. Its interval is spuriously narrow and should
   not be cited.
5. **Global search not tested.** Rastrigin and Ackley shifts lie in [−0.5, 0.5]ᵈ and
   every run starts at the centre, inside the global basin. These families test local
   convergence and basin retention, not global search.
6. **H1 control and H2 scope.**
   - The K=0 control also removes the rejection counter, so H1 cannot separate secant
     information from rejection-count information.
   - H2 tested cross-coordinate attention over history-enriched tokens, not temporal
     attention over pairs.
7. **Feature-schema weaknesses** found in review, left unchanged to preserve the
   pre-registration and assigned to a future `history-v2`:
   - Coordinates that never moved receive diagonal curvature 0, with no coverage mask.
   - Pair masks (`‖s‖∞ > 1e-7`, `‖y‖ > 1e-6‖g‖`) sit near float32 roundoff.
   - The legacy scale column intentionally breaks objective-scale invariance of the
     whole token: Rosenbrock replicates differing only in scale give different learned
     results.
8. **References are conservative.** L-BFGS-B received float32 JAX gradients cast to
   float64, and every call, including line-search trials, was charged two units.
9. **Statistics.** Three training seeds make the seed level of the bootstrap
   approximate. Win counts pair three seeds against one deterministic reference.
10. **Context arm.** The frozen v0.5 checkpoint was trained on Rastrigin and Ackley and
    for a different loop. It is context only.

11. **Training/inference equivalence coverage.**
    `test_training_unroll_matches_inference_loop_rule` and
    `test_training_scan_history_equals_inference_history` cover ordinary gradient
    scales only.
    - Below roughly `‖g_u‖ ≈ 1e-7`, training normalizes with `safe_l2_norm` (ε = 1e-8)
      while inference uses the exact norm and stops at a zero gradient.
    - When the objective decreases but the gradient is nonfinite, inference charges 2
      units and training charges 1.
    - These regimes are untested and the two loops may differ there.

## Corrections to statements in the frozen design

None of these changes the analysis.

- **Descent claim.** "P SPD ⇒ descent" holds before clipping. With a non-diagonal `P`,
  a clipped step can ascend; hard acceptance rejects it.
- **Inverse claim.** "Cannot equal `H_u⁻¹` for rotated Hessians in d>3" holds
  *generically*, not universally. The binding limit is the capacity cap in
  limitation 1.
- **Invariance claim.** "All are invariant to objective scaling" applies to the twelve
  history columns, not to the legacy scale column. The quadratic-consistency
  threshold is not invariant to shifting the objective.
- **Undocumented implementation details.** These were implemented before any run but
  are not spelled out in the design:
  - a `1e-6 Σ‖s‖‖y‖` guard in `ρ`;
  - a 1e-4 relative cancellation threshold for the quadratic-consistency feature;
  - "effective rank" computed as the participation ratio `n² / ‖ŜŜᵀ‖_F²`.
- **Metric.** The pre-registered metric (log10 normalized regret) replaces the
  `log1p(regret)` named in the draft experiment protocol.

## Decision under the stop rules

- **H1 passed.** History features may be developed further.
- **H2 failed.** No attention module is retained beyond the existing architecture.
  Temporal attention over pairs needs its own pre-registered round.
- **H3 failed.**
  - Fixed geometry stays the default.
  - Stochastic mass/mobility with history (G5) is not started.
  - Function-space and certification work (G6) is not started.

Recommended next round, pre-registered before any training:
- a `history-v2` schema with coverage masks and noise-aware pair reliability;
- a geometry parameterization able to express ≥100× anisotropy;
- a training unroll matched to the evaluation budget;
- BB, rather than the fixed step, as the bar a learned secant policy must clear.

## Reproduction

```bash
uv run python scripts/run_trajectory_curvature_v0_6.py --probe               # timing probe
uv run python scripts/run_trajectory_curvature_v0_6.py                       # train + evaluate
uv run python scripts/run_trajectory_curvature_v0_6.py --reuse-checkpoints   # evaluate only
uv run python scripts/run_trajectory_curvature_v0_6.py --tables              # tables above
```

- **Results:** `results/trajectory-curvature-v0.6.0.json`. It contains every row, the
  split hashes, sha256 of each source file, the design and every checkpoint, and the
  git HEAD (uncommitted tree).
- **Checkpoints:** `artifacts/trajectory-v0.6.0-<arm>-seed<s>.npz`.
- **Determinism:** re-evaluating the checkpoints after the review fixes reproduced all
  6,300 rows exactly.

# NeuralOpt v0.7 Track B: Gram-space history model, pre-registration

**Status:** draft for approval, 2026-11-22. **No training has been run.** Once
approved, this document is frozen. Any later change goes in a dated "deviations"
section of the results report, never into this text. Literature:
`docs/research/trajectory-curvature-literature.md` (v0.7 additions, verified).

## 1. Question

Can a small model that sees only rotation-invariant statistics of K = 10 actually
evaluated secant pairs produce an inverse metric P that beats an untrained
Barzilai–Borwein step on unseen rotated and ill-conditioned problems? The comparison
uses equal objective+gradient budgets and one frozen step-acceptance rule. A second
question is whether the model loses nothing to rotation.

## 2. Evidence this design rests on (already measured)

- **Gate 1, algebraic (passed).** With the BFGS coefficients
  (A = R⁻ᵀ, C = R⁻ᵀD^{1/2}/√τ, τe^δ = γ), P g equals the L-BFGS two-loop direction to
  within 1e-10. The coefficients lie inside the model's output ranges. P is SPD for
  every A and C, the tokens are invariant, and trajectories are rotation-equivariant
  and invariant to objective scale.
- **Anisotropy of at least 100× is demonstrated.** In the gate-2 runs the oracle's P
  reached condition numbers of 149 at κ = 100 and 1,363 at κ = 1000 (medians of
  per-run maxima). For comparison, the frozen v0.6 tf-h8 geometry is capped at about
  3.5× off-axis; measured runs reached at most 19.9.
- **Gate 2, capacity.** It FAILED as written. The amended criterion (regret floor
  1e-12; within one decade of SciPy maxcor 10) also fails at κ = 1000.
- **Loop-rule ablation.** The step-acceptance rule moves the oracle by at most 0.5
  decades, and history length matters more.
  - Frozen loop: **halving, K = 10**.
  - Halving is also the strongest rule for the untrained BB-like baseline, so the bar
    is not weakened.
- **Mask diagnostic** (`results/mask-diagnostic-v0.7.0.json`). The decision rule was
  stated before running.
  - Relaxed masks reduce the κ = 1000 median gap by **0.0 decades**, so by the rule they
    do **not** close a meaningful part of the gap.
  - Masking is rare: only the step test fires, on 3.4% of filled pairs at κ = 1000 and
    18% at κ = 100.
  - The gap is concentrated at d = 10, where the oracle reaches about −6 and SciPy −8.5
    to −9.8. At d = 7 both arms are at the float64 floor; at d = 16 both are near −5.
  - Paired over tasks at κ = 1000, oracle − SciPy is mean +0.86 [+0.18, +1.60] (raw),
    +0.69 [+0.08, +1.41] (floored at −12), and median +0.06.
  - **The remaining gap to SciPy at d = 10 is unexplained.** One untested hypothesis:
    with memory at least d, near-exact line minimisation gives quadratic termination,
    which unit steps forgo.

## 3. Capacity ceiling (stated, not a target)

The oracle uses BFGS coefficients and σ = 1 in the frozen loop (harness only), with
float64 masks.
- **κ = 100:** median −15.98, 12/12 reach 1e-6.
- **κ = 1000:** median −6.19, 7/12 reach 1e-6.

The learned model is not expected to exceed this ceiling. SciPy L-BFGS-B (maxcor 10)
reaches −15.96 and −8.79 on the same gate tasks. Its paired advantage at κ = 1000 is
stated above and remains unexplained.

## 4. Frozen design

- **Engine and rule.** Compiled engine, float64 throughout (features, training,
  evaluation). `loop_rule="halving"`: one trial; any finite decrease is accepted; a
  rejection halves the next trial. No Armijo test, and no BB-based base step inside the
  model: BB enters only as the τ scale and as a reference arm.
- **Model.** `model_kind="gram"`, K = 10 pairs, width 32, 4 heads, 2 layers (14,080
  parameters). Output ranges: |A|, |C| < 100; |δ| < 5; σ ∈ [1/4, 4]; ε_rel = 1e-12;
  fallback length 0.1.
- **history-v2 masks.** New schema `neuralopt-policy/v0.7-gram-v2`, with the thresholds
  stored in its config.
  - Thresholds: max|s| > 1e-12 and ‖y‖ > 1e-10·max‖g‖, plus finiteness.
  - **This is my call, flagged for approval.** The thresholds do not close the gap. I
    adopt them because the float32-era thresholds mask 15–18% of genuine pairs near
    convergence at κ = 100, and the protocol is float64. At κ = 1000 the ceiling is
    identical (−6.19) under either.
  - **Coverage.** Per-coordinate coverage has no rotation-invariant analogue. The tokens
    carry ŜᵀŜ (the rank and conditioning of the observed span), and P is isotropic on
    the unobserved complement by construction.
  - **Noise-aware reliability.** The tokens carry the cosines and norm ratios. No
    noisy-gradient family is in this round, so noise-aware reliability is **untested**.
- **Rejection information.** The rejection count, the last rejected step length and the
  ratio of observed to predicted change. Rejected trials never form pairs.
- **Training (implemented only after approval).**
  - The loop runs under `lax.scan` for the full budget of 100 units, at most 98
    candidates, with the inference budget masking. The training unroll therefore
    matches the evaluation budget, including the budget-fraction input.
  - A parity test against the compiled engine (trajectory and history to 1e-10) is
    required before any training.
  - Starts are random in the box.
  - Loss: mean over unit horizons {25, 50, 100} of log(normalized incumbent regret +
    1e-12), plus 0.1 × the mean candidate log-regret term.
  - Token gradients are stopped. The meta-gradient flows through positions and ∇f, so
    it is second-order; this cost is reported.
  - Adam 1e-3, global-norm clip 1.0, batch 8 tasks per update from one family and
    dimension group, 2,000 updates, final weights. There is no checkpoint selection and
    no tuning on any split.
  - NaN guard: a skipped update is counted and reported.
- **Seeds and cost.** Five training seeds (0–4) per trained arm.
  - Estimate, from the measured 97 µs per iteration of the Gram forward pass: about 98
    iterations × 4 (backward, second-order) × 97 µs ≈ 40 ms per task-gradient. That is
    16,000 task-gradients per run, about 11 minutes, single-threaded.
  - Stage 1: 2 arms × 5 seeds = 10 runs. Stage 2: 5 runs. About 2.8 CPU-hours in total,
    and about 15 minutes wall-clock in parallel.
  - A 20-update timing probe runs first. It may only lower the update count, uniformly
    for all arms, and only if one run would exceed 45 minutes.

## 5. Tasks and splits

Hashes are recorded. Test tasks are built only after training. Everything uses the
`make_task` generator.

- **Training.** Families: isotropic, diagonal κ ∈ {10, 100, 1000}, rotated
  κ ∈ {10, 100, 1000}, Rastrigin. Dimensions {4, 8, 12}, 4 tasks per family and
  dimension, seeds 7000+ (96 tasks). Objective scale cycles 0.2/1/5 and coordinate scale
  0.5/1/2.
- **Final test.** Seeds 9000+, unseen dimensions {7, 16, 32}, 4 tasks per family and
  dimension (72 tasks).
  - **Curvature domain** (primary): diagonal κ = 1000, rotated κ = 100, rotated
    κ = 1000.
  - **Equal-spectrum pairs:** the same seed gives the same permuted spectrum for
    diagonal and rotated κ = 1000.
  - **Context:** isotropic and Rastrigin.
  - **Held-out family:** Ackley.
  - Rosenbrock is excluded: its generator ignores the seed (pseudo-replication found in
    v0.6).
- **Budgets.** Primary 100 objective+gradient units; secondary 50 and 200.

## 6. Arms

| Arm | Kind | Trained | Stage |
|---|---|---|---|
| gram-scalar | Gram model with A = C = 0; learns δ and σ | 5 seeds | 1 |
| gram-nohist | same architecture, all pair tokens masked; keeps rejection information and the τ fallback | 5 seeds | 1 |
| gram-full | Gram model (A, C, δ, σ) | 5 seeds | 2 |
| BB | τ only (A = C = 0, δ = 0, σ = 1) in the frozen loop: **the bar** | no (harness reference) | 1, 2 |
| gram-oracle-BFGS | BFGS coefficients, σ = 1, frozen loop, float64 masks | no (harness only, never a package mode) | 2 |
| L-BFGS-B | SciPy, maxcor 10, 2 units per call, capped at 50 calls (line-search calls included) | no | context |
| tf-h8 | frozen v0.6 checkpoints, seeds 0–2, `history-v1`, frozen loop | no | context |

## 7. Metric and statistics

- **Per task:** D = log10(r_cand) − log10(r_ref), with r = max(f − f*, 0)/|f₀ − f*| and
  log10 r floored at −12. Raw values are reported as secondary. Negative favours the
  candidate.
- **Intervals.** Every contrast gets a 95% paired bootstrap interval with 10,000 draws.
  - Trained arm versus a reference: two-level crossed resampling of training seeds and
    tasks.
  - Reference versus reference: resampling over tasks only.
  - Five seeds make the seed level coarse; this is stated in the report.
- **Equivalence tests** use 90% intervals (two one-sided tests at 5%).

## 8. Hypotheses and stop rules

### Stage 1 (gram-scalar and gram-nohist only)

- **S1:** gram-scalar − BB on the curvature domain at budget 100.
  - **STOP before Stage 2 if the mean paired D is greater than 0**, that is, worse than
    BB on the point estimate, whatever the interval. The result is reported either way.
- **S1b (diagnostic, no stop):** gram-scalar − gram-nohist, which measures what the
  secant τ adds beyond rejection information. Also gram-nohist − BB.

### Stage 2 (only if S1 passes)

- **H1:** gram-full beats BB on the curvature domain: the upper 95% bound of D is below
  0.
  - **Reliability guard:** H1 is void if, on Rastrigin or Ackley, gram-full − BB has a
    mean above +0.5 with a lower bound above 0.
- **H2 (rotation).** For gram-full, the equal-spectrum difference rotated κ1000 −
  diagonal κ1000 (paired by seed) lies within ±0.5 decades: the 90% interval is inside
  (−0.5, +0.5). The same contrast is reported for BB, the oracle, L-BFGS-B and tf-h8
  as context; tf-h8 is expected to show a gap.
- **H3 (non-inferiority):** the upper 95% bound of gram-full − gram-oracle-BFGS is at
  most +0.5 decades on the curvature domain.
- **"Relearned L-BFGS" criterion (pre-stated).** If H3 holds **and** both of the
  following hold, the result is reported as *"gram-full relearned L-BFGS in this loop;
  no evidence of a new method"*:
  - (i) |gram-full − oracle| ≤ 0.25 decades on at least 75% of curvature-domain
    task × seed pairs;
  - (ii) on 200 fixed probe states (sampled from oracle trajectories on test tasks,
    seed 0), the median cosine between the model's and the oracle's step directions is
    at least 0.99.

  If instead the upper bound of gram-full − oracle is below 0, the result is reported as
  *"improves on L-BFGS directions in this loop"*, still with its comparison against
  SciPy L-BFGS-B. Otherwise neither statement is made.
- **Secondary (exploratory):** budgets 50 and 200; per-family and per-dimension results;
  held-out Ackley; all arms against L-BFGS-B.

## 9. Costs reported

- Objective, gradient and HVP counts (HVP is always 0), and units used.
- Accepted fraction, and the fraction of initial steps accepted unchanged.
- Cold, warm and batched wall-clock (Track A conventions, float64, single process).
- Training seconds per run, skipped updates, parameter counts, and checkpoint bytes and
  sha256.

## 10. Artifacts

- `artifacts/gram-v0.7.0-<arm>-seed<s>.npz` (Gram schema v2), with metadata recording
  split hashes and RNG.
- `results/gram-space-v0.7.0.json`.
- Frozen v0.1–v0.6 artifacts and results are never written.

## 11. Implementation still required after approval (before any training)

- The Gram schema v2 mask thresholds, with a schema bump and tests.
- The Gram training scan, with a parity test against the compiled engine.
- `train_batched` support for `model_kind="gram"`, including the NaN-skip test.
- The gram-nohist masking switch.
- An evaluation harness that reuses `accept_step` for the BB and oracle references,
  with parity checks.

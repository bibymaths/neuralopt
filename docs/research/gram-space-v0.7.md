# v0.7 Gram-space history model: pre-registered result

## Result

**The claim the frozen outcome-to-claim table permits: learned geometry did not beat BB
on the curvature domain; BB stays the bar.** H2 and H3 are exploratory, with no claim.

**This is a negative result.** Read off the frozen outcome-to-claim table
(pre-registration sha256 `6d5489da…3cad3f`), the row that applies is: Stage 1 passes,
H1 fails, so:

> *Learned geometry did not beat BB on the curvature domain; BB stays the bar. H2 and H3
> are exploratory, with no claim.*

- **Primary, H1, fails.** gram-full − BB on the curvature domain (diagonal κ1000,
  rotated κ100 and κ1000 at unseen d = 7/16/32, budget 100) is −0.224
  [−0.510, +0.115]. The interval is not entirely below 0.
- **The reliability guard is also violated.**
  - Rastrigin: +2.14 [+0.36, +4.34].
  - Ackley: +0.89 [+0.10, +2.08].
  - Shifted Rosenbrock: −0.06 [−0.11, −0.03].
  - Under the table, the guard changes nothing once H1 has failed. It is reported
    because it confirms the Stage 1 warning.
- **The learned model is far from relearning L-BFGS.**
  - gram-full − oracle-BFGS is +4.69 [+3.76, +5.68] decades.
  - Its step directions have a median cosine of 0.48 with the oracle's on 200 probe
    states.
  - Per family it is almost identical to the learned scalar step (gram-scalar): A and C
    added essentially nothing.
- **BB stays the bar,** and no claim is made for the Gram model. The headroom the oracle
  shows (A and C set to BFGS values) exists but was not learned.

## Stage 2: every pre-registered decision (`results/gram-space-stage2-v0.7.0.json`)

Intervals are 95% paired over 5 training seeds × tasks (two-level crossed bootstrap,
10,000 draws). Values are log10 regret floored at −12; negative favours the first arm.

| Decision | Contrast | Mean [95% CI] | Rule | Result |
|---|---|---|---|---|
| S1 (Stage 1, recomputed) | gram-scalar − BB, curvature | −0.139 [−0.334, +0.044] | upper < +0.25 | pass |
| **H1 (primary)** | gram-full − BB, curvature | **−0.224 [−0.510, +0.115]** | whole CI < 0 | **fail** |
| Guard | gram-full − BB, Rastrigin | +2.144 [+0.356, +4.338] | mean > +0.5 and lower > 0 | **violated** |
| Guard | gram-full − BB, Ackley | +0.892 [+0.095, +2.082] | same | **violated** |
| Guard | gram-full − BB, shifted Rosenbrock | −0.062 [−0.111, −0.026] | same | not violated |
| H2 (secondary; **exploratory** because H1 failed) | gram-full, rotated κ1000 − diagonal κ1000 (equal spectrum) | +0.054 [−0.257, +0.367] | whole CI within ±0.5 | within the margin; **no claim** |
| H3 (secondary; exploratory because H1 failed) | gram-full − oracle-BFGS, curvature | +4.688 [+3.758, +5.681] | upper < +0.5 | **fail** |
| R(i) | share of task × seed pairs with \|gram-full − oracle\| ≤ 0.25 | 0.000 | ≥ 0.75 | not met |
| R(ii) | median direction cosine with the oracle (200 probe states) | 0.482 | ≥ 0.99 | not met |

H2 context (rotated − diagonal at equal spectrum, reported only):
- BB: −0.037 [−0.379, +0.326]
- oracle: −0.254 [−0.589, +0.026]
- L-BFGS-B: +0.258 [+0.095, +0.438]
- tf-h8 (v0.6): +0.507 [+0.188, +0.828], the rotation penalty that v0.6's
  coordinate-axis geometry predicts.

## Per family at budget 100 (mean log10 regret, floored at −12; exploratory)

| Arm | diagonal-1000 | rotated-100 | rotated-1000 | isotropic | rastrigin | ackley | rosenbrock-shifted |
|---|---:|---:|---:|---:|---:|---:|---:|
| BB (bar) | −2.98 | −4.61 | −3.01 | −12.00 | −12.00 | −12.00 | −1.07 |
| oracle-BFGS (ceiling) | −7.14 | −10.80 | −7.40 | −12.00 | −12.00 | −10.26 | −2.90 |
| L-BFGS-B (context) | −7.55 | −10.96 | −7.29 | −12.00 | −10.19 | −7.15 | −1.99 |
| tf-h8 (v0.6, context) | −2.82 | −2.90 | −2.31 | −11.02 | −7.42 | −5.45 | −1.02 |
| gram-nohist | −2.17 | −2.36 | −1.99 | −11.83 | −7.98 | −7.54 | −0.99 |
| gram-scalar | −3.09 | −4.89 | −3.03 | −12.00 | −7.78 | −10.83 | −1.08 |
| **gram-full** | **−3.16** | **−5.02** | **−3.10** | **−12.00** | **−9.86** | **−11.11** | **−1.14** |
| gram-scalar, untrained (exploratory) | −2.62 | −3.57 | −2.62 | −12.00 | −11.67 | −10.76 | −1.05 |
| gram-full, untrained (exploratory) | −2.62 | −3.69 | −2.56 | −12.00 | −11.67 | −11.64 | −1.04 |

- **Direction cosine** (gram-full against the oracle, criterion R(ii)): median 0.48
  over 200 probe states from oracle trajectories on curvature-domain test tasks
  (seed-0 model).
- **Reliability:** no learned arm returned an error status. Mean units: 99.2–99.3 for
  the Gram arms and BB; 83.9 for L-BFGS-B, which stops early on some tasks.

## What training contributed (EXPLORATORY: not pre-registered, feeds no decision)

Source: `results/gram-untrained-diagnostic-v0.7.0.json`.
- **Why it was run:** requested because gram-scalar's training loss was flat from the
  start.
- **What it compares:** the same models at their initial weights
  (`init_gram_model(PRNGKey(seed))`, exactly what training starts from), on the same test
  tasks at budget 100, paired by seed.

| Arm | Trained − untrained, curvature | Rastrigin | Ackley | Shifted Rosenbrock | Untrained − BB, curvature |
|---|---|---|---|---|---|
| gram-scalar | −0.736 [−1.020, −0.513] | +3.893 [+0.392, +7.885] | −0.075 [−1.003, +0.907] | −0.030 [−0.058, −0.007] | +0.598 [+0.325, +0.886] |
| gram-full | −0.805 [−1.072, −0.562] | +1.814 [−0.239, +4.218] | +0.533 [−0.270, +1.675] | −0.095 [−0.153, −0.050] | +0.581 [+0.311, +0.873] |

- **Training helped on the curvature domain.** It moved both arms from about 0.6
  decades worse than BB to a tie.
- **Training hurt on Rastrigin.** The untrained models stay in the global basin, about
  as BB does; the trained ones do not.
- **The flat training loss hides this.** The gain on the curvature domain and the
  regression on Rastrigin both happened during training.
- These are exploratory observations, not claims.

## Post-mortem (EXPLORATORY: not pre-registered, no new claims, feeds no decision)

Source: `results/gram-postmortem-v0.7.0.json`.
- **Hypothesis tested:** the model configuration (depth and width) limits learning the A
  and C coefficients.
- **Data:** only training tasks and validation tasks (seeds 8500+, dimensions 6 and 10).
  No test task was built. Frozen v0.7 results and checkpoints are unchanged.
- **Oracle states:** oracle-BFGS runs in the frozen loop (halving, K = 10, history-v2
  masks, float64, budget 100) gave 6,505 training states and 2,101 validation states.

### 1. Imitation sweep

Each configuration was fitted by supervised regression to the oracle's A, C·Cᵀ (C is
defined only up to a right orthogonal factor), δ and σ = 1. Training used 3,000 Adam
steps, batch 256 and seed 0. Heads = width/8, so the head dimension is 8 as in the
frozen model.

**Fit columns:** median relative error on held-out validation states, and the median
cosine between the fitted model's step direction and the oracle's.

**Regret column:** median log10 regret when the fitted model runs the shared loop on the
12 curvature-domain validation tasks.

**Reference on those 12 tasks:**
- BB −3.32;
- oracle −15.99;
- the frozen meta-trained gram-full, per seed: −3.82, −3.34, −3.94, −3.57, −3.67.

| Layers × width | Parameters | Rel. error A | Rel. error CCᵀ | \|δ error\| | Direction cosine | Regret (validation) |
|---|---:|---:|---:|---:|---:|---:|
| 2 × 32 (frozen config) | 14,080 | 0.700 | 0.731 | 0.089 | 0.916 | −2.34 |
| 2 × 64 | 52,736 | 0.435 | 0.488 | 0.088 | 0.947 | −2.60 |
| 2 × 128 | 203,776 | 0.349 | 0.466 | 0.049 | 0.963 | −1.79 |
| 4 × 32 | 22,272 | 0.630 | 0.662 | 0.080 | 0.922 | −2.24 |
| 4 × 64 | 85,504 | 0.444 | 0.541 | 0.071 | 0.937 | −2.55 |
| 4 × 128 | 334,848 | 0.328 | 0.430 | 0.067 | 0.970 | −2.47 |
| 6 × 32 | 30,464 | 0.708 | 0.765 | 0.109 | 0.891 | −2.20 |
| 6 × 64 | 118,272 | 0.485 | 0.535 | 0.074 | 0.956 | −2.37 |
| 6 × 128 | 465,920 | 0.339 | 0.438 | 0.043 | 0.966 | −2.70 |

### 2. Head saddle check

**Setup:**
- 200 updates of the frozen meta-training on training tasks, with the same batching as
  `train_batched`.
- The loop is verified to reproduce `train_batched` exactly for seed 0: maximum
  parameter difference 0.0.
- Seeds 0, 1 and 2, under two initializations: the frozen one (A/C head weights at
  0.1 scale), and a variant with those weights rescaled to 1.0 scale.
- No NaN skips occurred.

**Measures:**
- Median per-update gradient norm (before the global-norm clip at 1.0) at the A, C, δ
  and σ heads.
- A/C output magnitudes (mean Frobenius norm) on 256 fixed validation probe states.
- For scale: on training states with pairs, the oracle targets have median ‖A*‖_F 4.5
  and median ‖C*C*ᵀ‖_F 5.0.

| Init | Seed | ‖∇A heads‖ | ‖∇C heads‖ | ‖∇δ‖ | ‖∇σ‖ | (A,C) / (δ,σ) ratio | ‖A‖_F: 0 → 200 | ‖C‖_F: 0 → 200 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 0.1 (frozen) | 0 | 22.8 | 3.3 | 137 | 39.1 | 0.14 | 0.010 → 0.034 | 0.010 → 0.249 |
| 0.1 (frozen) | 1 | 35.4 | 11.0 | 179 | 53.1 | 0.19 | 0.009 → 0.260 | 0.007 → 0.495 |
| 0.1 (frozen) | 2 | 53.3 | 1.4 | 245 | 71.4 | 0.18 | 0.008 → 0.089 | 0.009 → 0.084 |
| 1.0 | 0 | 293 | 92.1 | 678 | 245 | 0.44 | 0.966 → 1.573 | 1.014 → 0.497 |
| 1.0 | 1 | 158 | 66.3 | 293 | 120 | 0.57 | 0.850 → 0.513 | 0.724 → 0.390 |
| 1.0 | 2 | 89.7 | 65.7 | 193 | 92.8 | 0.55 | 0.807 → 0.744 | 0.912 → 0.922 |

### 3. Conclusion (exploratory)

**Depth and width are not the binding limit.**
- Width improves imitation fit: relative A error falls from 0.70 to 0.33–0.35, and the
  direction cosine rises from 0.92 to 0.96–0.97. Depth adds little.
- But no fitted configuration performs well in the shared loop. Every one is worse than
  BB on validation (−1.79 to −2.70, against −3.32), and worse than the frozen
  meta-trained model (−3.3 to −3.9).
- Even direction cosines of 0.96–0.97 with the oracle do not reproduce the oracle's
  −15.99. That points to the loop's sensitivity to small metric errors and to the shift
  between oracle-visited and self-visited states. Neither cause was isolated.

**There is no head saddle.**
- Gradients reach the A and C heads from the frozen init, and A/C magnitudes grow 3–30×
  in 200 updates.
- They stay 20–150× below the oracle's median ‖A*‖_F of 4.5.
- The A/C gradient is only about 0.14–0.19 of the δ/σ gradient, and every update is
  dominated by δ: total norms of 135–370 are clipped to 1.0.
- With init scale 1.0, A and C do not collapse, but their share stays at 0.44–0.57,
  and δ moves most (|δ| 0.3 to 1.1–1.6).

**Overall, the failure is undetermined, leaning towards optimization and objective
signal rather than representation.** The meta-gradient mostly drives the scalar step
(δ/σ). Larger networks fit the oracle better without performing better in the loop.
Which change would let A and C be learned (the loss, the gradient balance or clipping,
on-policy imitation, or more updates) was not tested. Any such change belongs to a
future v0.8 pre-registration, not to v0.7.

## Decision and next work

- **By plan gate G7, the trajectory-curvature line is not expanded further.** The
  validation-tuned fixed diagonal and the BB step remain the reference geometries, and
  no v0.7 learned geometry becomes a default.
- **The next work is learned guidance for certified global search.** This was set by
  the user on 2026-11-22. In this repository it connects to plan gate G6 (optional
  feasible-incumbent candidate injection into certified search), which the v0.6 plan
  deferred until learned geometry showed value. Certification itself remains
  independent and model-free.
- Any changed model or training setup for the Gram line, including the post-mortem
  ideas above, needs a new pre-registration (v0.8). It is not added to v0.7.

## Training and compute

- **Stage 2 training:** gram-full, 5 seeds × 2,000 updates, locked configuration.
  - **0 NaN-skipped updates.**
  - 9.0 minutes wall-clock in parallel; 8.7–8.9 minutes per run; 17.8–18.5 minutes of
    process CPU per run (including XLA helper threads); **1.51 CPU-hours** in total.
  - Loss (mean natural-log regret per 100 updates) ranged from −10.8 to −13.2 in the
    first 100 updates and ended at −12.5 to −13.2, with occasional excursions (seed 4,
    updates 1400–1500: −10.8).
- **Total for Stages 1 and 2:** 15 runs, **4.32 CPU-hours** against the probe's 5.0
  CPU-hour estimate, and 18.2 minutes wall-clock of training.
- **Evaluation:** Stage 1 219 s; Stage 2 296 s; untrained diagnostic 69 s.
- **Hashes:** the sha256 of all 15 checkpoints and of the pre-registration are in
  `results/gram-space-stage2-v0.7.0.json`.

## Stage 1 results (frozen protocol; `results/gram-space-stage1-v0.7.0.json`)

### Training
- gram-scalar and gram-nohist, 5 seeds each, 2,000 updates.
- **0 NaN-skipped updates** in all 10 runs.
- 9.2 minutes wall-clock in parallel; 2.81 CPU-hours.
- Per run: 7–9 minutes wall-clock and 15–19 minutes of process CPU time (the CPU time
  includes XLA helper threads).
- Checkpoints: `artifacts/gram-v0.7.0-gram-{scalar,nohist}-seed{0..4}.npz`, with sha256
  in the results file.
- Training loss (mean log regret per 100 updates):
  - gram-scalar: flat from the start, about −12.1 to −13.1 at first, ending at −12.6 to
    −13.5.
  - gram-nohist: improves noisily, from −5.5 to −7.3 at first to −7.2 to −9.1 at the end.

### Stage 1 decision
Curvature domain, budget 100, two-level paired intervals (5 seeds × 36 tasks).

| Contrast | Mean D [95% CI] | Rule |
|---|---|---|
| **S1: gram-scalar − BB** | **−0.139 [−0.334, +0.044]** | upper bound < +0.25: **pass, continue to Stage 2** |
| S1b: gram-scalar − gram-nohist | −1.499 [−1.844, −1.173] | diagnostic |
| S1b: gram-nohist − BB | +1.360 [+1.084, +1.647] | diagnostic |

### Per family (exploratory; mean floored log10 regret at budget 100)

| Arm | diagonal-1000 | rotated-100 | rotated-1000 | isotropic | rastrigin | ackley | rosenbrock-shifted |
|---|---:|---:|---:|---:|---:|---:|---:|
| BB | −2.98 | −4.61 | −3.01 | −12.00 | −12.00 | −12.00 | −1.07 |
| oracle-BFGS | −7.14 | −10.80 | −7.40 | −12.00 | −12.00 | −10.26 | −2.90 |
| L-BFGS-B | −7.55 | −10.96 | −7.29 | −12.00 | −10.19 | −7.15 | −1.99 |
| tf-h8 (v0.6) | −2.82 | −2.90 | −2.31 | −11.02 | −7.42 | −5.45 | −1.02 |
| gram-scalar | −3.09 | −4.89 | −3.03 | −12.00 | −7.78 | −10.83 | −1.08 |
| gram-nohist | −2.17 | −2.36 | −1.99 | −11.83 | −7.98 | −7.54 | −0.99 |

**gram-scalar − BB on the guard families (exploratory):**
- Rastrigin: **+4.22 [+1.08, +7.96]**
- Ackley: **+1.17 [+0.50, +1.85]**
- shifted Rosenbrock: −0.00 [−0.02, +0.01]

No error statuses. Mean units: 99.2 for the Gram arms; 83.9 for L-BFGS-B, which stops
early on some tasks.

### Reading (no claims beyond the frozen rules)

- **The secant scale carries the information.** The learned scalar step ties BB on the
  curvature domain. Without secant pairs, with only rejection information, the same
  architecture is 1.36 decades worse than BB and 1.50 decades worse than gram-scalar.
- **The learned scalar step regresses on Rastrigin and Ackley,** where BB stays in the
  global basin. Stage 1 has no guard rule, so this is exploratory. **It is an early
  warning for the Stage 2 reliability guard,** which applies to gram-full with the same
  thresholds. Nothing was changed in response.
- **On the unseen test dimensions (7, 16, 32), the oracle ceiling is close to SciPy:**
  rotated-100 −10.80 against −10.96, rotated-1000 −7.40 against −7.29.


## Pre-registration (frozen) and infrastructure

- **Frozen pre-registration:**
  `docs/superpowers/specs/2026-11-22-gram-space-v07-preregistration.md`, sha256
  `6d5489dab06d3bbc5b60f716ab65411e8626af65f3a97de0a0044e24b43cad3f`. The hash was
  recorded in its own commit, and `scripts/run_gram_space_v0_7.py` checks it at startup.
- **Infrastructure built after freezing (no training run):**
  - Gram schema v2 (`neuralopt-policy/v0.7-gram-v2`), with history-v2 masks and the
    `history_masked` no-history switch. v1 files load with v1 thresholds.
  - The matched training unroll (`neuralopt.gram_training`, a 97-step scan at budget
    100). It equals the compiled engine to 1e-10 for full, scalar and no-history
    variants.
  - `train_batched(model_kind="gram")`, with the NaN-skip test.
  - The reference and evaluation harness (`scripts/run_gram_space_v0_7.py`): BB and
    oracle through `accept_step`, a parity check against the package loop, SciPy, the
    tf-h8 context arm, two-level paired intervals, the decision rules and the
    outcome-to-claim table, all tested in `tests/test_gram_space_protocol.py`.
- **Timing probe** (`results/gram-probe-v0.7.0.json`; gram-full, 20 updates, nothing
  saved):
  - 0.59 s per update; 13.5 task-gradients/s; about 5.5 s for the first pass including
    the 6 compilations.
  - Projected 19.9 minutes per run single-process, under the 45-minute limit, so the
    **2,000 updates are kept**.
  - 15 runs: **about 5.0 CPU-hours**, or about 20 minutes wall-clock per stage when the
    runs execute in parallel.
  - The draft's earlier estimate of about 11 minutes per run was low by roughly 2×.


## Development history: model, gates, loop rule and masks (before the pre-registration)

### The model (B1; opt-in, schema `neuralopt-policy/v0.7-gram-v1`)

Source: `neuralopt.gram`. It runs only through
`minimize(..., model_kind="gram", candidate_method="preconditioned", preconditioned_engine="compiled")`.

- **Tokens.** There is one token per stored secant pair plus one for the current
  gradient: K+1 tokens, independent of the dimension d. Every input is a normalized
  inner product among the columns of S, Y and g_u, or a log norm ratio, mask, pair age,
  rejection count, last rejected step length, or observed/predicted change ratio of the
  last rejected trial. Rejected trials never form secant pairs.
- **Outputs.** Masked attention produces A and C (K×K), δ and σ. They define
  `P = τ[e^δ (I − ŜAŶᵀ)(I − ŶAᵀŜᵀ) + ŜCCᵀŜᵀ] + ετI`, applied without forming the
  matrix in O(Kd).
  - τ = sᵀs/sᵀy of the newest valid pair when sᵀy > 0, otherwise
    `fallback_length/‖g_u‖`.
  - The step is `u' = clip(u − shrink·σ·P g_u, −1, 1)`, with the same acceptance and
    accounting as the shared loop.
- **What P is.** P is symmetric positive definite for every A and C, and it is an
  *inverse* metric (P y ≈ s). Its directional structure is confined to span(Ŝ, Ŷ); on the
  complement it is isotropic.

### Gate 1: algebraic, float64 (`tests/test_gram.py`)

All checks pass.

- **(a) The model can express L-BFGS.** With every valid pair having sᵀy > 0, set
  - τe^δ = γ = sᵀy/yᵀy of the newest pair,
  - A = R⁻ᵀ and C = R⁻ᵀD^{1/2}/√τ, with Rᵢⱼ = sᵢᵀyⱼ for i ≤ j (chronological) and
    D = diag(sᵢᵀyᵢ),
  - converted to normalized columns as Â = D_s A D_y and Ĉ = D_s C.

  Then P g equals the L-BFGS two-loop direction with H₀ = γI to within 1e-10. This holds
  for d/K = 10/4, 10/6, 16/8 and 5/5, and P y = s holds on the newest pair. The
  coefficients lie inside the model's output ranges. This is the compact representation
  of Byrd, Nocedal & Schnabel (1994); the independent derivation review found no error.
- **(b) SPD.** P is symmetric positive definite for random A and C (minimum eigenvalue at
  least ετ). The matrix-free product equals the dense matrix. Masked pairs, rows and
  columns, contribute nothing.
- **(c) Invariance.** Tokens are invariant to rotation of u and to positive rescaling of
  the objective. Whole trajectories rotate with a rotated quadratic: a ±1000 cube at the
  origin with an interior optimum, so clipping is never active. Trajectories are
  unchanged by rescaling the objective, which rescales τ, while σ is dimensionless.

### Gate 2: capacity, no training (`results/gram-gates-v0.7.0.json`)

**Criterion, stated before the first run:** for both rotated κ = 100 and κ = 1000, the
oracle-BFGS arm's median log10 regret must be within one decade of SciPy L-BFGS-B on the
same tasks.

Conditions: d ∈ {7, 10, 16}, 4 tasks per d, gate-only seeds 6000+, budget 100
objective+gradient units, float64. The oracle sets A, C and δ to the BFGS values and
σ = 1. It lives only in the harness, `scripts/gram_reference.py` and
`scripts/run_gram_gates_v0_7.py`. The harness copy of the loop reproduces the package's
compiled Gram loop to 1e-10 over 37 iterations; this is checked before any oracle number
is reported.

| | oracle-BFGS | oracle clipped to the model's ranges | τ only (A=C=0, δ=0, σ=1) | L-BFGS-B | oracle − L-BFGS-B | Pass |
|---|---:|---:|---:|---:|---:|---|
| rotated κ=100, median log10 regret | −13.80 | −13.80 | −5.23 | −15.96 | +2.16 | **no** |
| rotated κ=1000, median log10 regret | −5.49 | −5.49 | −2.72 | −8.79 | +3.30 | **no** |

Further measurements:
- **Reached 1e-6 regret:** at κ=100, oracle 12/12 and L-BFGS-B 12/12; at κ=1000,
  oracle 5/12 and L-BFGS-B 8/12.
- **Acceptance:** the oracle accepted 89% of trials at κ=100 and 93% at κ=1000, so
  rejections are rare.
- **Conditioning:** the oracle's P reached a median maximum condition number of 149 at
  κ=100 and 1,363 at κ=1000. The Gram parameterization therefore expresses at least 100×
  anisotropy.
- **Coefficient sizes:** the oracle coefficients never left the model's output ranges.
  The maximum |Â| was 13.4 and |Ĉ| 7.8, against a bound of 100; |δ| reached 4.84,
  against a bound of 5. Clipping to the model's ranges changes nothing.

#### Reading

- **κ=100.** Both arms reach 1e-6 on every task. The 2.2-decade gap lies at relative
  regret around 1e-14 to 1e-16, the float64 roundoff zone. Under the pre-stated rule
  this still counts as a failure, and it is reported as one.
- **κ=1000.** The 3.3-decade gap is substantive.
- **Where the limit is.** The same BFGS directions, run through this loop rule, fall well
  short of L-BFGS-B at equal budget. The loop accepts any decrease with a unit step and
  only halves after a rejection. There is no sufficient-decrease or curvature (Wolfe)
  condition and no step lengthening.
- **Untested explanations:** the accepted steps may produce poorer secant pairs than
  Wolfe line-search steps do, and K = 8 against SciPy's default memory of 10 may also
  matter.
- **Rejection signal under clipping.** With a non-diagonal P, a clipped step can ascend
  for every shrink factor. This is a loop-rule property, shared with v0.6, and it makes
  the rejection signal ambiguous near active bounds. These gate tasks have interior
  optima.

### Gate 2 as written, and the amended criterion

- **As written, gate 2 FAILED and stays failed.**
- At κ=100 both arms are at the float64 floor, and the original criterion had no
  regret floor.
- For the reruns below, the user stated an **amended criterion** before any rerun was
  made:
  - per task, log10 regret below −12 counts as −12, so regrets below 1e-12 are equal;
  - an arm passes iff its floored median is within one decade of SciPy L-BFGS-B's
    floored median, at both κ=100 and κ=1000.

### Loop-rule ablation (`results/loop-rule-ablation-v0.7.0.json`)

#### Setup
- Same gate tasks as gate 2, float64, budget 100 objective+gradient units.
- Every arm steps through `neuralopt.loop_rules.accept_step`, so all arms share one
  step-acceptance rule. Only the BFGS oracle's *direction* lives in the harness.
- The harness outer loop reproduces the package's compiled engine (fixed geometry) to
  1e-10 under each rule before any number is reported.
- "Initial unchanged" means the model's own step was accepted without correction.

#### The three rules
- **halving (v0.6):** one trial per iteration; any decrease is accepted; a rejection
  halves the next trial.
- **armijo:** sufficient decrease with interpolating backtracking. Each trial costs 1
  objective unit, and the accepted point costs 1 gradient unit.
- **wolfe:** strong Wolfe conditions with bracketing and zoom, lengthening up to 10×.
  Each trial costs 2 units.

#### Results

| Arm | κ=100 median (floored) | 1e-6 | κ=1000 median | 1e-6 | Units | Accepted / iterations (κ=1000) | Initial unchanged (κ=1000) | Amended pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| oracle K=8, halving | −13.80 (−12) | 12/12 | −5.49 | 5/12 | ≤100 | 0.91 | 0.86 | no |
| oracle K=8, armijo | −13.71 (−12) | 12/12 | −5.70 | 5/12 | ≤100 | 1.00 | 0.92 | no |
| oracle K=8, wolfe | −13.63 (−12) | 12/12 | −5.59 | 5/12 | 100 | 1.00 | 0.92 | no |
| oracle K=10, halving | −14.47 (−12) | 12/12 | −6.19 | 7/12 | ≤100 | 0.92 | 0.87 | no |
| oracle K=10, armijo (supplementary) | −14.96 (−12) | 12/12 | −6.68 | 7/12 | ≤100 | — | 0.92 | no |
| oracle K=10, wolfe (supplementary) | −14.22 (−12) | 12/12 | −6.43 | 7/12 | 100 | — | 0.93 | no |
| τ only, halving | −5.23 | 2/12 | −2.72 | 0/12 | ≤100 | 0.58 | 0.43 | no |
| τ only, armijo | −3.52 | 0/12 | −2.61 | 0/12 | ≤100 | 0.99 | 0.63 | no |
| τ only, wolfe | −3.32 | 0/12 | −2.32 | 0/12 | 100 | 0.99 | 0.57 | no |
| fixed step, halving | −3.16 | 0/12 | −2.19 | 0/12 | ≤100 | 0.64 | 0.37 | no |
| fixed step, armijo | −3.26 | 0/12 | −2.31 | 0/12 | ≤100 | 0.99 | 0.60 | no |
| fixed step, wolfe | −3.22 | 0/12 | −2.18 | 0/12 | 100 | 0.98 | 0.62 | no |
| SciPy L-BFGS-B, maxcor 10 | −15.96 (−12) | 12/12 | −8.79 | 8/12 | 100 | — | — | reference |
| SciPy L-BFGS-B, maxcor 8 | −15.14 (−12) | 12/12 | −5.98 | 5/12 | 100 | — | — | — |

"Supplementary" rows were not part of the requested ablation. They separate history
length from the loop rule.

#### Reading

- **The step-acceptance rule is not the bottleneck for good directions.**
  - Under all three rules the oracle's own step is accepted unchanged 86–93% of the time
    at κ=1000.
  - Changing the rule moves the oracle by at most 0.5 decades.
- **For the no-learning baselines the rule matters, in the other direction.** At κ=100,
  τ-only does best under halving (−5.23, against −3.52 and −3.32).
- **History length matters a lot.**
  - SciPy loses 2.8 decades at κ=1000 when its memory drops from 10 to 8.
  - At matched memory 8, every oracle loop is within 0.3–0.5 decades of SciPy.
  - At K=10, a gap of 2.1–2.6 decades to SciPy (maxcor 10) remains under every rule.
    The cause is not identified.
- **The amended criterion fails.** Every arm is at the floor at κ=100, but at κ=1000
  none comes within one decade of SciPy L-BFGS-B at its default memory.
- **Uncertainty.** Medians over 12 tasks are coarse, and no paired intervals were
  computed for this ablation.

#### Mask diagnostic (oracle K=10, halving; `results/mask-diagnostic-v0.7.0.json`)

The decision rule was stated before running: relaxed masks count as meaningful only if
they reduce the κ=1000 median gap to SciPy (maxcor 10) by at least one decade.
- **Masking is rare.** Under the current thresholds (step 1e-7, relative gradient
  change 1e-6), only the step test fires: on 3.4% of filled pairs at κ=1000 and 18% at
  κ=100. The gradient-change and nonfinite tests never fire. On average 8.6 of 10 pairs
  are valid at κ=1000.
- **Relaxed masks change nothing at κ=1000.** With float64 thresholds (1e-12 and
  1e-10), or masking only nonfinite pairs and pairs with sᵀy ≤ 0, the oracle median
  stays at −6.19. **The reduction is 0.0 decades, so not meaningful.**
- **At κ=100 relaxed masks help,** moving the median from −14.47 to −15.98, but that
  is at the float64 floor.
- **Paired over tasks at κ=1000,** oracle − SciPy is mean +0.86 [+0.18, +1.60] (raw)
  and +0.69 [+0.08, +1.41] (floored at −12), with median +0.06.
- **The gap sits at d=10:** the oracle reaches about −6 and SciPy −8.5 to −9.8. At d=7
  both are at the floor; at d=16 both are near −5.
- **The cause is unexplained.**

#### Step-acceptance rules (scope)

`loop_rule` is a step-acceptance rule in the sense of CLAUDE.md, not a solver.
- It decides whether, and how far, to move along the model's proposed displacement.
- The model's step is always the first trial.
- No curvature is built: there is no BFGS update and no classical solver.
- It is opt-in, on the compiled engine only (`loop_rule="halving" | "armijo" |
  "wolfe"`, default `"halving"`).
- The Python loop, the v0.6 rule and all frozen results are unchanged.

### tf-h8 anisotropy cap (the frozen v0.6 checkpoints)

- With `geometry_bound = 10` and `lowrank_scale = 0.5`, the diagonal ranges over
  [0.1, 10] and λmax(UUᵀ) < 0.25.
- So the cap is 100× along the coordinate axes and about **3.5× off-axis**.
- Along tf-h8 runs (3 seeds × rotated κ=1000 × d = 7/10/16), the bound
  cond(P) ≤ (diag_max + ‖U‖_F²)/diag_min reached a maximum of 19.9 and a median of 3.3.

### Review findings and dispositions

Source: curvature-reviewer, 2026-11-22.

- **Fixed:**
  - Gram diagnostics now record τ and σ in their own fields, with a test.
  - The oracle arm now reports coefficient sizes, and a correctly clipped
    bounded-oracle arm was added.
  - The gate (a) test now checks that the coefficients lie inside the model's ranges.
  - The compiled engine's docstring now describes the Gram rule.
  - A separate fix from this round: the compiled engine casts the budget fraction to the
    parameters' dtype, which removes a future-JAX type error for float32 checkpoints on
    float64 problems.
- **Open, for the next design decision:**
  - The pair mask uses a max-norm, so it is not exactly rotation-invariant near its
    1e-7 threshold.
  - The default ε is 1e-12, which provides no positive-definiteness floor in float32.
  - A near-noise newest pair can make τ very large in float32.
  - The oracle handles all-valid histories; masking is tested only for the oldest slots.
  - Any change to the token layout must bump `GRAM_SCHEMA`.
  - The clip-induced ascent described under "Rejection signal under clipping".


## Corrections

These are deviations from the frozen pre-registration, or details it did not fix. The
frozen file is never edited.

1. **Training group sampling** (fixed before any training). The groups (quadratic-d4/8/12
   with 28 tasks each, Rastrigin-d4/8/12 with 4 each) are cycled in proportion to their
   size, 7:1, so every training task is equally likely. Uniform cycling would have put
   50% of updates on Rastrigin.
2. **Reporting-only harness additions** (committed before Stage 1 training): per-run
   loss curves and CPU time; checkpoint sha256 in the results; exploratory per-family
   tables; error-status counts. No decision, threshold or claim changed.
3. **The first Stage 1 launch failed** before training: `/usr/bin/time` is not
   installed. Nothing was trained; the relaunch timed itself with `date` plus each run's
   process CPU time.
4. **The Stage 2 evaluation re-evaluates the Stage 1 arms** on the same frozen tasks
   (deterministic). It reproduces the Stage 1 decision exactly:
   −0.139 [−0.334, +0.044].
5. **Untrained-weights diagnostic and post-mortem** (added after Stage 2, at the user's
   request): both are **exploratory, not pre-registered, and feed no decision.** They
   used training and validation tasks only, except that the untrained diagnostic used
   the already-built test tasks.
6. **CPU accounting:** process CPU time includes XLA helper threads, so it is about
   twice the per-run wall time. The probe's estimate was a single-process wall-time
   projection.
7. **A test was changed before training** so that it does not build test-split tasks.
   The equal-spectrum property is now checked on seeds outside all splits.

## Limitations

- **Statistics.** Five training seeds make the seed level of the bootstrap coarse.
  Medians and floored means at −12 hide differences below 1e-12.
- **Scope.** Exact gradients and float64 only; noise-aware pair reliability is untested.
  The test dimensions are 7, 16 and 32 for a model trained at 4, 8 and 12.
- **The oracle's gap to SciPy at d = 10** (gate tasks) remains unexplained. At the test
  dimensions the oracle is close to SciPy.
- **BB here** means τ-only (sᵀs/sᵀy of the newest valid pair) inside the frozen halving
  loop. It is not SciPy and not v0.6's harness-only BB variant.
- **tf-h8 is context only:** a v0.6 history-v1 model with float32 checkpoints, run here
  through the compiled engine on float64 problems.
- **Untested hypotheses.** Why A and C were not learned (their training signal, the
  loss, the 2,000 updates or the model size) is not established. Neither is why the
  trained models leave the Rastrigin and Ackley basins. Neither was investigated, and
  nothing was tuned after the results.

## Reproduction

```bash
uv run python scripts/run_gram_gates_v0_7.py            # gates (harness only)
uv run python scripts/run_loop_rule_ablation_v0_7.py    # loop-rule ablation
uv run python scripts/run_mask_diagnostic_v0_7.py       # mask diagnostic
uv run python scripts/run_ceiling_d32_v0_7.py           # d=32 ceiling (validation seeds)
uv run python scripts/run_gram_space_v0_7.py --probe
uv run python scripts/run_gram_space_v0_7.py --stage 1 && uv run python scripts/run_gram_space_v0_7.py --evaluate 1
uv run python scripts/run_gram_space_v0_7.py --stage 2 && uv run python scripts/run_gram_space_v0_7.py --evaluate 2
uv run python scripts/run_gram_untrained_diagnostic_v0_7.py   # exploratory
uv run python scripts/run_gram_postmortem_v0_7.py              # exploratory post-mortem
```

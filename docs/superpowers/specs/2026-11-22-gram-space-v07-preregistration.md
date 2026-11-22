# NeuralOpt v0.7 Track B: Gram-space history model, pre-registration

**Status:** FROZEN on 2026-11-22. The approved draft was revised as instructed. This
file's sha256 is recorded in the commit that follows it and in
`docs/superpowers/specs/2026-11-22-gram-space-v07-preregistration.sha256`. **No
training had been run when it was frozen.** Any later deviation goes under
"Corrections" in the results report (`docs/research/gram-space-v0.7.md`), never into
this file. Literature: `docs/research/trajectory-curvature-literature.md` (v0.7
additions, verified against OpenAlex).

## 1. Question

Can a small model that sees only rotation-invariant statistics of K = 10 actually
evaluated secant pairs produce an inverse metric P that beats an untrained
Barzilai–Borwein (BB) step on unseen rotated and ill-conditioned problems? The
comparison uses equal objective+gradient budgets and one frozen step-acceptance rule.

## 2. Evidence this design rests on (measured before freezing)

- **Gate 1, algebraic (passed).** With the BFGS coefficients
  (A = R⁻ᵀ, C = R⁻ᵀD^{1/2}/√τ, τe^δ = γ), P g equals the L-BFGS two-loop direction to
  within 1e-10, and the coefficients lie inside the model's output ranges. P is SPD for
  every A and C, the tokens are invariant, and trajectories are rotation-equivariant and
  invariant to objective scale.
- **Anisotropy of at least 100× is demonstrated.** In gate 2 the oracle's P reached
  condition numbers of 149 at κ = 100 and 1,363 at κ = 1000 (medians of per-run
  maxima). The frozen v0.6 tf-h8 geometry is capped at about 3.5× off-axis (at most
  19.9 measured).
- **Gate 2.** It FAILED as written. The amended criterion (regret floor 1e-12; within
  one decade of SciPy maxcor 10) also fails at κ = 1000.
- **Loop-rule ablation.** The step-acceptance rule moves the oracle by at most 0.5
  decades.
  - Frozen loop: **halving, K = 10**.
  - Halving is the strongest rule for the untrained BB-like baseline, so the bar is not
    weakened.
- **Mask diagnostic.** The decision rule was stated before running.
  - Relaxed masks reduce the κ = 1000 median gap by 0.0 decades, so they are not
    meaningful for the gap.
  - Paired oracle − SciPy at κ = 1000 (d = 7/10/16): mean +0.86 [+0.18, +1.60], median
    +0.06.
  - The gap sits at d = 10 and is **unexplained**.

## 3. Capacity ceiling (stated, not a target)

The oracle uses BFGS coefficients and σ = 1 in the frozen loop with history-v2 masks
(harness only). The table gives median log10 regret and the number of tasks reaching
1e-6.

| Tasks | κ = 100 | κ = 1000 | Paired oracle − SciPy maxcor 10 |
|---|---:|---:|---|
| gate tasks, d = 7/10/16 (12 per κ) | −15.98 (12/12) | −6.19 (7/12) | κ=1000: +0.86 [+0.18, +1.60] (gap at d = 10, unexplained) |
| **validation seeds 8000+, d = 32 (4 per κ)** | **−9.30 (4/4)** | **−5.07 (0/4)** | κ=100: −0.34 [−0.47, −0.14]; κ=1000: −0.02 [−0.23, +0.20] |

The d = 32 values come from `results/oracle-ceiling-d32-v0.7.0.json`, measured on
validation seeds, not test tasks. **At d = 32 the oracle matches or beats SciPy
L-BFGS-B.** The learned model is not expected to exceed this ceiling.

## 4. Frozen design

- **Engine and rule.** Compiled engine, float64 throughout (features, training,
  evaluation). `loop_rule="halving"`: one trial; any finite decrease is accepted; a
  rejection halves the next trial. No Armijo test, and no BB-based base step inside the
  model: BB enters only as the τ scale and as a reference arm.
- **Model.** `model_kind="gram"`, K = 10 pairs, width 32, 4 heads, 2 layers (14,080
  parameters). Output ranges: |A|, |C| < 100; |δ| < 5; σ ∈ [1/4, 4]; ε_rel = 1e-12;
  fallback length 0.1.
- **history-v2 masks** (approved). Schema `neuralopt-policy/v0.7-gram-v2`, with the
  thresholds stored in its config.
  - Thresholds: max|s| > 1e-12 and ‖y‖ > 1e-10·max‖g‖, plus finiteness.
  - Reasons: the protocol is float64; the masks are neutral at κ = 1000, reach the floor
    at κ = 100, and were chosen before training.
  - **Coverage.** The tokens carry ŜᵀŜ (rank and conditioning of the observed span); P
    is isotropic on the unobserved complement by construction.
  - **Noise-aware reliability is untested:** no noisy-gradient family is in this round.
- **Rejection information.** The rejection count, the last rejected step length and the
  ratio of observed to predicted change. Rejected trials never form pairs.
- **No-history control** (`history_masked=True`). Every pair is masked, so the model
  sees only the gradient token (rejection information and budget fraction) and τ uses
  its fallback.
- **The training unroll is matched to the evaluation budget of 100 units.**
  - The training loop is the inference loop under `lax.scan`: the same `accept_step`
    halving rule, the same budget accounting (start costs 2 units; a trial costs 1,
    plus 1 if it improved), and the same stopping condition (`used + 2 ≤ 100` and a
    nonzero gradient). Inactive scan steps are masked.
  - The scan length is the maximum number of iterations at budget 100, which is
    **97**. The budget-fraction input is `1 − used/100`, as in inference.
  - A parity test against the compiled engine is required before any training:
    trajectory, history and counts must match to 1e-10.
- **Training settings.**
  - Starts are random in the box.
  - Loss: mean over unit horizons {25, 50, 100} of
    log(normalized incumbent regret + 1e-12), plus 0.1 × the mean over active steps of
    candidate log-regret.
  - The history buffer (S, Y, f) is treated as observed data: `stop_gradient` applies
    in both the tokens and the metric. The meta-gradient flows through the current
    position and ∇f, so it is second-order; this cost is reported.
  - Adam 1e-3, global-norm clip 1.0, batch 8 tasks per update from one family and
    dimension group, 2,000 updates, final weights. There is no checkpoint selection and
    no tuning on any split.
  - NaN guard: a skipped update is counted and reported.
- **Seeds and cost.** Five training seeds (0–4) per trained arm.
  - The cost estimate is in the infrastructure report.
  - A 20-update timing probe runs first. It may only lower the update count, uniformly
    for all arms, and only if one run would exceed 45 minutes.

## 5. Tasks and splits

Hashes are recorded. Test tasks are built only after training. Everything uses the
`make_task` generator.

- **Training.** Families: isotropic, diagonal κ ∈ {10, 100, 1000}, rotated
  κ ∈ {10, 100, 1000}, Rastrigin. Dimensions {4, 8, 12}, 4 tasks per family and
  dimension, seeds 7000+ (96 tasks). Objective scale cycles 0.2/1/5 and coordinate scale
  0.5/1/2.
- **Validation seeds 8000+.** Used only for the d = 32 ceiling. There is no tuning and
  no selection.
- **Final test.** Seeds 9000+, unseen dimensions {7, 16, 32}, 4 tasks per family and
  dimension (84 tasks).
  - **Curvature domain** (primary): diagonal κ = 1000, rotated κ = 100, rotated
    κ = 1000.
  - **Equal-spectrum pairs:** the same seed gives the same permuted spectrum for
    diagonal and rotated κ = 1000.
  - **Context:** isotropic.
  - **Reliability-guard families:** Rastrigin (in-distribution), and Ackley and
    shifted Rosenbrock (held out).
- **Shifted Rosenbrock** (`rosenbrock-shifted`) replaces v0.6's seed-independent
  Rosenbrock. Each replicate has a seeded shift in [−0.5, 0.5]ᵈ, giving a distinct
  interior minimizer c(1 + shift). This is tested.
- **Budgets.** Primary 100 objective+gradient units; secondary 50 and 200.

## 6. Arms

| Arm | Kind | Trained | Stage |
|---|---|---|---|
| gram-scalar | Gram model with A = C = 0; learns δ and σ | 5 seeds | 1 |
| gram-nohist | same architecture with `history_masked`; keeps rejection information and the τ fallback | 5 seeds | 1 |
| gram-full | Gram model (A, C, δ, σ) | 5 seeds | 2 |
| BB | τ only (A = C = 0, δ = 0, σ = 1) in the frozen loop with history-v2 masks: **the bar** | no (harness reference) | 1, 2 |
| gram-oracle-BFGS | BFGS coefficients, σ = 1, frozen loop, history-v2 masks | no (harness only, never a package mode) | 2 |
| L-BFGS-B | SciPy, maxcor 10, 2 units per call, capped at 50 calls (line-search calls included) | no | context |
| tf-h8 | frozen v0.6 checkpoints, seeds 0–2, `history-v1`, frozen loop | no | context |

## 7. Metric, statistics and decision rules

- **Per task:** D = log10(r_cand) − log10(r_ref), with r = max(f − f*, 0)/|f₀ − f*| and
  log10 r floored at −12. Raw values are reported as secondary. Negative favours the
  candidate.
- **Intervals.** Every contrast gets a 95% paired bootstrap interval with 10,000 draws.
  - Trained arm versus a reference: two-level crossed resampling of training seeds and
    tasks.
  - Reference versus reference: resampling over tasks only.
  - Five seeds make the seed level coarse; this is stated in the report.
- **Decision rules** (all on the 95% paired interval):
  - **Superiority:** the whole interval is below 0.
  - **Non-inferiority at margin m:** the upper bound is below +m.
  - **Equivalence:** the whole interval lies within ±0.5 decades.

## 8. Hypotheses and stop rules

### Stage 1 (gram-scalar and gram-nohist only)

- **S1 (non-inferiority):** gram-scalar − BB on the curvature domain at budget 100.
  - **STOP before Stage 2 only if the upper bound is above +0.25 decades.**
  - Reason (recorded): the oracle shows that the headroom over BB comes from the
    metric (A, C), not the scalar step. A tie at Stage 1 therefore should not end the
    round.
- **S1b (diagnostic, no stop):** gram-scalar − gram-nohist, which measures what the
  secant τ adds beyond rejection information. Also gram-nohist − BB.

### Stage 2 (only if S1 passes)

- **Primary, H1:** gram-full beats BB on the curvature domain at budget 100, meaning
  the whole interval is below 0.
- **Reliability guard (G):** H1 is void if, on any of Rastrigin, Ackley or shifted
  Rosenbrock, gram-full − BB has a mean above +0.5 with a lower bound above 0.
- **Secondary, H2 (rotation equivalence):** for gram-full, the equal-spectrum difference
  rotated κ1000 − diagonal κ1000 (paired by seed) lies within ±0.5 decades over its
  whole interval. The same contrast is reported for BB, the oracle, L-BFGS-B and tf-h8
  as context.
- **Secondary, H3 (oracle non-inferiority):** the upper bound of gram-full −
  gram-oracle-BFGS on the curvature domain is below +0.5.
- **"Relearned L-BFGS" criterion (R, pre-stated, evaluated only if H3 holds).** R is met
  if both hold:
  - (i) |gram-full − oracle| ≤ 0.25 decades on at least 75% of curvature-domain
    task × seed pairs;
  - (ii) on 200 fixed probe states (sampled from oracle trajectories on test tasks,
    seed 0), the median cosine between the model's and the oracle's step directions is
    at least 0.99.
- **Exploratory:** budgets 50 and 200; per-family and per-dimension results; all arms
  against L-BFGS-B.

### Outcome-to-claim table

Claims are cumulative within a row. "No claim" means the result is reported but nothing
is asserted. Under no outcome is a claim made of superiority over SciPy L-BFGS-B, or of
novelty over L-SR1, Optimus, LODO or the v0.7 literature.

| Stage 1 S1 | H1 | Guard G | H3 | R | H2 | What we may claim |
|---|---|---|---|---|---|---|
| fails (upper > +0.25) | — | — | — | — | — | The learned scalar step is worse than BB beyond the 0.25-decade margin. The round ends; BB stays the bar; no Stage 2. |
| passes | fails | any | any | any | any | Learned geometry did **not** beat BB on the curvature domain. BB stays the bar. H2 and H3 are reported as exploratory, with no claim. |
| passes | passes | violated | any | any | any | **No superiority claim:** the curvature-domain gain comes with a reliability regression on a multimodal family. Both are reported as exploratory. |
| passes | passes | holds | fails | — | passes | Beats BB on unseen rotated and ill-conditioned tasks at equal budget, with no measurable rotation penalty (±0.5). It remains short of the oracle-BFGS ceiling. |
| passes | passes | holds | fails | — | fails | Beats BB at equal budget, short of the oracle ceiling. **No rotation-invariance claim;** the rotated − diagonal gap is reported. |
| passes | passes | holds | passes | met | passes | Beats BB with no measurable rotation penalty, **by relearning L-BFGS-like directions in this loop: no evidence of a new method.** |
| passes | passes | holds | passes | met | fails | Beats BB by relearning L-BFGS-like directions (no new method). No rotation-invariance claim. |
| passes | passes | holds | passes | not met | passes | Beats BB with no measurable rotation penalty, and is non-inferior to oracle-BFGS (+0.5) without matching its directions. Whether it differs usefully from L-BFGS is not established. |
| passes | passes | holds | passes | not met | fails | Beats BB and is non-inferior to the oracle. No rotation-invariance claim. |

If the upper bound of gram-full − oracle is below 0 (better than the oracle), that is
reported as "improves on L-BFGS directions in this loop". It is still reported against
SciPy L-BFGS-B, with no claim over it.

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

## 11. Infrastructure built after freezing (before any training)

- The Gram schema v2 mask thresholds and the `history_masked` switch, with a schema
  bump and tests.
- The Gram training scan (the matched unroll), with a parity test against the compiled
  engine.
- `train_batched` support for `model_kind="gram"`, including the NaN-skip test.
- An evaluation and reference harness that reuses `accept_step`, with parity checks and
  the statistics above. A 20-update timing probe is run; no training.

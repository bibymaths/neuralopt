# NeuralOpt v0.6 — trajectory-aware curvature: design and pre-registration

**Status:** frozen before any v0.6 training or evaluation (2026-11-21). Changes after
results are seen must be listed in a dated "deviations" section of the results report,
never silently edited here.

## Question and why

v0.5 found learned stochastic geometry worse than a validation-tuned fixed scalar
geometry on unseen dimensions, and diagnosed missing information: one gradient cannot
identify rotated curvature. v0.6 asks whether a small variable-dimensional policy that
sees 4–8 secant pairs `(s, y)` from **actually evaluated** states produces geometry that
beats the tuned fixed baseline on unseen rotated / ill-conditioned objectives, under
exactly matched objective+gradient counts, with runtime and training cost reported.

## Goals / non-goals / anti-goals

- Goal: an opt-in, versioned history feature schema and a shared deterministic
  evaluation loop; a pre-registered multi-seed ablation; an honest report.
- Non-goals (this round): temporal attention over pairs, stochastic mass/mobility with
  history (G5), function-space or certification injection (G6), HVP probes.
- Anti-goals: any change to legacy outputs or checkpoint loading; classical solvers in
  package inference; uncounted evaluations; tuning on final test tasks; claims beyond
  the measured evidence.

## Approaches considered (direct-mode arena)

- **A (chosen by user):** new opt-in loop `candidate_method="preconditioned"` shared by
  every arm, fixed baseline = same loop with scalar geometry, references in harness.
- B: legacy direct loop + history features — rejected: proposal head consumes half the
  evaluations and rejected steps are re-proposed almost identically (v0.2 diagnosis).
- C: stochastic arena first — rejected: contradicts "stochastic only after gates".

## Mathematics (normalized box coordinates)

`u = 2(x-lo)/(hi-lo) - 1`, `J = diag((hi-lo)/2)`, `g_u = J g_x`, `H_u = J H_x J`.
For consecutive history entries `k, k+1` (both actually evaluated with objective and
gradient at the same point): `s_k = u_{k+1}-u_k`, `y_k = g_{u,k+1}-g_{u,k}`; on a
quadratic `y_k = H_u s_k` exactly (up to rounding), whatever produced `s_k`.
K pairs identify `H_u` only on `span(S)` (rank ≤ min(K, d)); two Hessians that agree on
`span(S)` give identical histories, so no full-Hessian recovery is claimed. The geometry
`P = diag(d) + U Uᵀ` (rank 2) stays SPD; it acts as an **inverse** metric (ideal
`P y ≈ s`), cannot represent indefinite curvature, and cannot equal `H_u⁻¹` for rotated
Hessians in d > 3. Negative `sᵀy` is kept as a feature, never inverted.

### History state (`src/neuralopt/trajectory.py`)

`TrajectoryHistory(u: (K+1,d), g: (K+1,d), f: (K+1,), valid: (K+1,) bool,
rejections: ())` — shift buffer, index `K` newest, explicit chronological order, O(Kd).
Entries are pushed **only** for accepted, evaluated states (start point included).
Rejected candidates increment `rejections`; they have no gradient and never form pairs.

Pair mask `m_k = valid_k ∧ valid_{k+1} ∧ finite(s_k,y_k) ∧ ‖s_k‖∞ > 1e-7 ∧
‖y_k‖ > 1e-6·max(‖g_k‖,‖g_{k+1}‖)` (degenerate / roundoff-level pairs are masked).

### Feature schema `history-v1` (17 per coordinate token)

Legacy five (`parameter_features(..., scale_aware=True)`) followed by twelve history
features. With `c_k = s_kᵀy_k`, `σ_k = s_kᵀs_k`, `R_k = c_k/σ_k`, `cos_k = c_k/(‖s_k‖‖y_k‖)`,
`n = Σm_k`, self-scaling `ρ = Σ m|c_k| / Σ mσ_k`:

| # | feature | kind |
|---|---|---|
| 6 | `asinh(a_i/ρ)/4`, `a_i = Σ m s_ki y_ki / (Σ m s_ki² + 1e-12 Σ mσ_k)` (diagonal secant curvature, signed) | coord |
| 7 | newest valid `s_i/‖s‖` | coord |
| 8 | newest valid `y_i/‖y‖` | coord |
| 9 | `tanh(log(‖g_u‖/ρ)/4)` (log Newton step length in u units) | global |
| 10 | fraction of valid pairs with `c_k < 0` | global |
| 11 | `n / K` | global |
| 12 | mean valid `|cos_k|` (reliability / conditioning) | global |
| 13 | `tanh(log(max|R_k| / min|R_k|)/4)`, 0 if n < 2 | global |
| 14 | effective rank of normalized valid `S` divided by K | global |
| 15 | `1[n > 0]` | global |
| 16 | `tanh(rejections/4)` | global |
| 17 | `tanh(mean_k m_k clip(Δf_k / (½(g_k+g_{k+1})ᵀs_k) - 1, ±10))` (quadratic-model consistency, 0 on quadratics) | global |

Features 6–10, 12–14, 17 are exactly zero when `n = 0`, with 11 and 15 as explicit masks.
All are invariant to objective scaling and to box rescaling, and coordinate-permutation
equivariant (globals invariant). `K = 0` gives the matched no-history control with
identical width and parameter count.

### Model and checkpoint schema

`ModelConfig` gains `feature_set: "legacy" | "history-v1" = "legacy"` and
`history_length: int = 0` (must be 0 for legacy) and property `input_features`
(5 or 17). Init uses `input_features` for `embed` / MLP `hidden`. Legacy configs save
byte-compatible files (no new keys). Non-legacy configs add key
`schema = "neuralopt-policy/v0.6-history-v1"` and two extra config entries; loading an
unknown schema raises. `_validate_policy_params` checks the embedding input width.

### Shared evaluation loop (`candidate_method="preconditioned"`)

State: accepted `x`, its `f`, `g_x` (each evaluated once), shrink `σ=1`, history.
Each iteration: `ĝ = g_u/‖g_u‖`; geometry `P` = learned (model, structure flag
honoured), `fixed` (`fixed_metric_diag`, U=0) or identity; step `L` = `0.5·step_head`
(learned) or `preconditioned_step` (non-learned, required). Candidate
`u' = clip(u − σ L P ĝ, −1, 1)`; one objective call; accept iff finite and `f' < f`, then
one gradient call, push history, `σ=1`; else `σ ← σ/2`, `rejections += 1`.
Budget counts objective **plus** gradient calls (start point costs 2); iterate while
`used + 2 ≤ budget`. HVP count is 0 and reported. The direction head is unused
(P SPD ⇒ descent). Legacy loops are untouched; stochastic methods reject history configs.

### Meta-training (`objective="history"`)

`history_regret_loss` mirrors the loop in `lax.scan` (T = 24, hard acceptance via
`where`, gradient at the candidate computed for training only). Loss
`mean_h log(r_h + 1e-6) + 0.1·mean_t log(r^cand_t + 1e-6)` with
`r = max(f − f*, 0)/(|f₀ − f*| + 1e-12)`, horizons (6, 12, 24), features
`stop_gradient`ed (history is treated as observed data), random start in the box,
Adam 1e-3, global-norm clip 1.0. Identical for every trained arm.

## Pre-registered protocol

- **Tasks** (generator `scaled_task` copied from v0.5, objective scale cycle
  0.2/1/5, coordinate scale 0.5/1/2): train families isotropic, diagonal κ∈{10,100,1000},
  rotated κ∈{10,100}, rastrigin; dims 2/4/8; seeds 1000+ (one per variant×dim, 21
  tasks). Validation: same variants, dims 3/6, seeds 2000+ (14 tasks). **Final test**
  (seeds 3000+, dims 7/10/16, 4 seeds per family×dim, 84 tasks): in-distribution
  isotropic, diagonal κ1000, rotated κ100, rastrigin; held-out families rotated κ1000,
  rosenbrock, ackley. Curvature domain = diagonal κ1000 + rotated κ100 + rotated κ1000
  (36 tasks).
- **Arms** (all in the shared loop, budget 100 units, start at `problem.x0`):
  `fixed` (step L ∈ {0.01, 0.03, 0.1, 0.3, 1} chosen on validation only);
  `bb` harness reference (L = min(‖g_u‖/R_last, 1) when the newest valid R > 0, else
  tuned fixed L); `lbfgs` SciPy L-BFGS-B harness reference (each call = 1 obj + 1 grad,
  `maxfun = 50`); trained `tf-legacy` (v0.5 architecture, 5 features); `mlp-h0/4/8`
  (width 48) and `tf-h0/4/8` (width 16, 4 heads, 1 layer, rank 2,
  `geometry_bound=10`, `lowrank_scale=0.5`, `scale_aware=True`); contextual zero-shot
  frozen `v0.5 overdamped_full`. Three training seeds {0, 1, 2}; final-update weights
  (no checkpoint selection). Updates per arm: 400 (fixed now; a timing probe may only
  lower it uniformly if one arm exceeds 15 min CPU).
- **Metric:** `D = log10(r_cand + 1e-10) − log10(r_ref + 1e-10)` per task (and
  training seed); negative favours the candidate. Two-level bootstrap (training seeds,
  then tasks), 10,000 draws, 95% percentile CI.
- **Primary candidate:** `tf-h8`. **H1** (history value): `tf-h8 − tf-h0` on the
  curvature domain, supported iff CI upper < 0. **H2** (attention value):
  `tf-h8 − mlp-h8`, same rule. **H3** (scientific): `tf-h8 − fixed`, supported iff
  CI upper < 0 and no test family has mean D > +0.5 with CI lower > 0. Everything else
  (K=4, MLP, references, budgets 50/200, per-family, held-out families) is secondary
  and exploratory.
- **Stop rules:** H1 fails ⇒ no temporal attention, no G5; report and stop. H3 fails ⇒
  fixed geometry stays the default. Temporal attention and G5 each need a new
  pre-registered round regardless of outcome.
- **Also reported:** obj/grad/HVP counts, acceptance, nonfinite rate, first-call and
  warm runtime, training seconds, parameter counts, checkpoint bytes.
- **Artifacts:** `artifacts/trajectory-v0.6.0-<arm>-seed<s>.npz` (schema, split hashes,
  RNG provenance in metadata), `results/trajectory-curvature-v0.6.0.json`. v0.5
  artifacts are never written.
- **Resource budget:** ≤ 3 h CPU wall-clock training, ≤ 1 h evaluation (96 cores, no GPU).

## Left to the builder

Internal helper names, diagnostic record fields beyond the counts above, script CLI flags.

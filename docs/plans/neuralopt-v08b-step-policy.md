# NeuralOpt v0.8-B: a learned step policy in its plausible regime (plan)

**Status: approved on 2026-11-22, with the decisions in §14.** Implementation still waits
for the main track's gate F0 (stop point 2) to be reported and for the user's go-ahead, so
v0.8-B builds on the verified families, owned L-BFGS and baselines. No training or heavy
compute happens before then.

This plan expands section 15 of `neuralopt-v08-amortized-families.md`. It keeps the v0.8
release freeze (CLAUDE.md rule 8), the same Families 1 and 2b with their hashed splits, the
same accounting conventions (§6 of the main plan), and has **its own pre-registration**.
Claims C11 and C12 are in `docs/research/v0.8-claims.md` as `unverified`; their criteria
below are proposed and are frozen only at this track's pre-registration.

## 1. Question and hypotheses

v0.4–v0.7 tested learned steps on final precision at moderate budgets; fixed diagonal or BB
steps won. v0.8-B asks whether a learned step policy wins where such policies are most
plausible: **multimodal or valley-shaped problems, at 20–100 evaluations, judged on anytime
progress and on reaching the global basin**, not on final precision.

- **H11 (C11):** on a family, the policy's anytime regret (§5.1) is lower than **every**
  baseline's at every pre-registered budget, under unit matching and under wall-clock
  matching.
- **H12 (C12):** on a family, the policy reaches the global basin (§5.2) with a higher
  probability than **every** baseline, at every pre-registered budget, under both matchings.

Both are falsifiable per family: either can come out supported, not supported or
inconclusive (§6), and every outcome is reported.

## 2. Families and splits

All families are `f(x; c)` on a box, float64, d = 4 in distribution.

| Family | Definition | Box | Instance distribution | Global optimum |
|---|---|---|---|---|
| Rastrigin (rotated, shifted) | `z = R(x − o)`; `f = A·d + Σ (z_j² − A cos 2π z_j)` | `[−5.12, 5.12]^d` | `o ~ U[−2.5, 2.5]^d`, `R` Haar-orthogonal, `A ~ U[5, 15]` | `x = o`, `f* = 0` |
| Ackley (rotated, shifted) | `z = R(x − o)`; standard Ackley with a = 20, b = 0.2, c = 2π | `[−32.768, 32.768]^d` | `o ~ U[−16, 16]^d`, `R` Haar-orthogonal | `x = o`, `f* = 0` |
| Shifted Rosenbrock | `z = x − o + 1`; `f = Σ 100 (z_{j+1} − z_j²)² + (1 − z_j)²` | `[−5, 5]^d` | `o ~ U[−2.5, 2.5]^d` | `x = o`, `f* = 0` (a second local minimum exists for 4 ≤ d ≤ 7) |
| Family 1 (damped sinusoid) | main plan §8, `neuralopt.families` "sinusoid" | as defined there | as defined there | reference optimum `f*` (main plan §6) |
| Family 2b (realistic cascade) | main plan §8, "cascade-realistic" | as defined there | as defined there | reference optimum `f*` |

- Rotations are in distribution on purpose. Unrotated Rastrigin and Ackley are separable,
  which favours per-coordinate step policies; a rotated test removes that advantage.
- **Splits:** train 2,000, validation 200, test 200, and 200 per OOD shift. Instance `i` of
  split `s` is seeded by `sha256(family|split|i)`, as in `neuralopt.families`; every split
  has a sha256 manifest. Families 1 and 2b reuse their existing splits and hashes.
- **OOD shifts (reported, not part of the C11/C12 criteria):**
  - synthetic families: d = 8; shifts outside the training range (still inside the box);
    Rastrigin `A ∈ [15, 25]`;
  - Families 1 and 2b: their existing OOD splits.
- **First evaluation point.** Every arm's first evaluation on instance `i` is the same point
  `x0_i`: drawn uniformly in the box (instance-seeded) for the synthetic families, and the
  family default for Families 1 and 2b (as F0's A0). Multistart arms take their further
  starts from a fixed Sobol sequence.

## 3. Arms

### 3.1 The policy (the learned arm)

- **Inputs,** all in box-normalized coordinates `u = 2(x − lo)/(hi − lo) − 1`, with
  gradients mapped as `g_u = Jᵀ g_x`, `J = diag((hi − lo)/2)`:
  - per coordinate and per step, over a window of the last H = 16 evaluated states: `u_j`,
    `g_u,j` divided by the window's RMS gradient, and the secant differences `Δu_j`,
    `Δg_j`, masked when invalid or rejected (never filled with zeros);
  - per step: the normalized objective relative to the best so far, the step index and the
    remaining budget fraction;
  - Families 1 and 2b only: an instance-context embedding from the start predictor's masked
    set encoder. The synthetic families get no context, because their only context (shift
    and rotation) would reveal the optimum.
- **Architecture:** POP-style (Kobiolka et al. 2026). A small transformer attends over time
  per coordinate (weights shared across coordinates), followed by one cross-coordinate
  mixing layer so that rotated problems can be represented. Target size 20k–50k
  parameters.
- **Outputs:** per-coordinate mean and log standard deviation of a log step size,
  `η_t ~ N(μ_t, σ_t²)`, sampled for exploration (as in OPEN; Goldie et al. 2024). The
  update is `u_{t+1} = Π_box(u_t − exp(η_t) ⊙ g̃_t)`.
  - Optional ablation: an Optimus-style low-rank term (rank 2) added to the update (Gärtner
    et al. 2023). It is off in the primary arm.
- **Evaluation randomness:** the sampling RNG is seeded per instance, so evaluation is
  reproducible.
- **Cost:** one objective and one gradient call at each new point (2 units).
- **Checkpoints** use a new versioned schema (`step-policy-v1`); the history-v1 and Gram
  schemas and all v0.1–v0.7 checkpoints are untouched.
- Every step at inference comes from NeuralOpt's own code (CLAUDE.md).

### 3.2 Baselines

Each is tuned on validation only, per family and per budget, over the grid below; the
chosen configuration is frozen in the pre-registration.

| Baseline | Implementation | Tuning grid (validation) |
|---|---|---|
| Owned multistart L-BFGS | `neuralopt.lbfgs` (pure JAX) | K ∈ {1, 2, 4, 8} starts, B/K units each |
| SciPy multistart L-BFGS-B | SciPy, harness only | K ∈ {1, 2, 4, 8} |
| CMA-ES | `cma` (pycma), harness only; mean at `x0_i`, in `u` coordinates, box bounds | σ0 ∈ {0.1, 0.2, 0.3} of the `u` box width; popsize ∈ {default, 2× default} |
| TPE | `optuna.samplers.TPESampler`, in-memory storage, harness only; first trial enqueued at `x0_i` | n_startup_trials ∈ {5, 10}; multivariate ∈ {False, True} |
| BB | NeuralOpt's compiled-loop BB step, projected onto the box | initial step ∈ {1e-3, 1e-2, 1e-1}; step clip ∈ {1, 10} |
| Quasi-random search | scrambled Sobol points in the box, the first at `x0_i`; harness only | none |

**Quasi-random search is in the "every baseline" set** (decision 1). At 20–100 evaluations
it is the natural floor; a policy that beat the other baselines but lost to sampling would
not support C11 or C12.

Stochastic baselines (CMA-ES, TPE, quasi-random search with its scramble seed) are run with
5 seeds; each instance's outcome is the mean over seeds, as for the policy.

## 4. Cost accounting and matching

- **Units.** One objective call = 1 unit; one gradient call = 1 unit; HVPs are not used.
  Objective and gradient calls are **reported separately** for every arm.
  - Owned L-BFGS: an objective per line-search trial and a gradient on acceptance (§3 of
    the main plan).
  - SciPy, BB and the policy: a joint value-and-gradient call costs 2 units. Its value
    becomes available at the second unit.
  - **CMA-ES, TPE and random search use no gradients.** At budget B they make B objective
    calls, and their gradient count is zero.
- **Unit matching undercharges gradients.** F0's probe measured the gradient at 2.5× an
  objective call for Family 1 and about 12× for Family 2. Unit matching therefore favours
  gradient-based arms. This is why **every claim must also hold under wall-clock
  matching**. A cost-weighted count `n_f + r·n_g`, with the measured ratio `r`, is reported
  as an exploratory column.
- **Wall-clock matching (warm).**
  - Each arm's median warm cumulative-time curve `C_a(t)` (seconds after `t` units) is
    measured per family on 50 validation instances. Measurement is one instance at a time
    (the deployment case), after compilation, on the same CPU, float64, with XLA
    single-threaded. The policy's forward passes are included. The curves are frozen in
    the pre-registration.
  - The wall budget for budget B is `T(B) = C_policy(B)`. Baseline `k` gets
    `B_k = max{t : C_k(t) ≤ T(B)}` units and is tuned on validation at `B_k`.
  - Wall-matched anytime regret integrates `L` (§5.1) over time `[0, T(B)]`, mapping each
    arm's units to time through its frozen `C_a`. This keeps the comparison deterministic.
  - Actual test timings are recorded too. A deviation from the frozen curves above 25% is
    reported.
- **Cold time** (including compilation) and batched throughput are reported separately and
  are not part of the criteria.
- **Not charged to any arm:** the global-basin polish (§5.2) and the regret normalization
  scale `s_i` (§5.1). Both are measurement only and never feed back into an arm.
- **Charged as offline cost and reported:** training (all generations, including
  compilation) and the training-split references.

## 5. Metric definitions

### 5.1 Anytime regret: the area under the regret curve

For instance `i`:

- `f_best(t)` is the lowest objective value among the arm's objective evaluations completed
  within the first `t` units. It is undefined before the first value.
- **Floor:** `τ_i = ε_abs + ε_rel·|f*_i|`, the success tolerance, with ε as in F0
  (ε_abs = 1e-6, ε_rel = 1e-3). Precision below the success tolerance earns nothing, so the
  metric does not reward final precision.
- **Normalization:** `s_i` is the median of `f(x) − f*_i` over 64 fixed Sobol points in the
  box (harness only, uncharged). It is the regret of a typical uninformed point.
  Instances with `s_i ≤ τ_i` are excluded and counted.
- **Clipped log regret:**
  `L_i(t) = log10( min(max(f_best(t) − f*_i, τ_i), 10·s_i) / s_i )`, and `L_i(t) = 1` before
  the first value.
  - Its range is `[log10(τ_i/s_i), 1]`.
  - If any arm finds a value below `f*_i − τ_i`, then `f*_i` is updated for all arms and
    the update is reported, as in F0.
- **Anytime regret at budget B:** `AUC_i(B) = (1/B) Σ_{t=1..B} L_i(t)`. **Lower is better.**
  The wall-matched version is `(1/T) ∫_0^T L_i(τ) dτ`, as in §4.
- **Budget grid:** B ∈ {20, 50, 100} units.
  - Baselines are run separately at each B, with their B-tuned configuration.
  - The policy takes the remaining budget as an input and is also run separately at each B.
  - Full curves for t = 1..100 are reported for every arm.

### 5.2 Global basin

- **Definition (operational):** an arm *reached the global basin* at budget B if a
  measurement-only polish started from its best point `x_best(B)` converges to within the
  success tolerance of the global optimum: `f(polish(x_best)) − f*_i ≤ τ_i`.
  - The polish is NeuralOpt's owned L-BFGS, capped at 2,000 units with gtol 1e-8 (the
    reference cap), with no restarts. It is uncharged and never fed back.
  - The global basin is thus the basin of attraction of a fixed local polish, not a
    geometric object.
- **Per family:**
  - Rastrigin, Ackley, shifted Rosenbrock: `f* = 0` at `x = o` (known).
  - Families 1 and 2b: `f*` is the reference optimum from the main plan's §6 protocol
    (SciPy multistart, 64 Sobol starts plus the truth, two seeds, ≥ 99% two-seed
    reproduction on validation), updated if any arm beats it.
- **Sanity rows (reported):** the rate from polishing `x0_i` directly (zero search), and
  the rate from polishing the best of B Sobol points.

## 6. How C11 and C12 are decided

- **Unit of analysis:** test instances, paired across arms. The policy's outcome for an
  instance is the mean over **≥ 5 training seeds**. Stochastic baselines are averaged over
  5 run seeds.
- **Differences:**
  - C11: `d_i = AUC_i(policy) − AUC_i(baseline)`; negative is better for the policy.
  - C12: `d_i = p_i(policy) − p_i(baseline)`, where `p_i` is the share of seeds reaching
    the global basin; positive is better for the policy.
- **Intervals:** a two-level percentile bootstrap resamples training seeds, then instances
  (paired). It uses 10,000 resamples and a fixed RNG seed.
  - Each comparison uses a **two-sided 95% interval**.
- **"Beats all baselines" is an intersection-union test** (Berger 1982), decided **per
  family.** The claim holds for a family only if *every* comparison is individually
  significant: 6 baselines × 3 budgets × 2 matchings = **36 comparisons**.
  - Because the claim is a conjunction, its error rate is at most the level of each
    comparison: a one-sided 2.5% chance of a false "supported" for the family.
  - No multiplicity correction is applied within a family, and none to the per-family
    statuses in the register.
- **Joint claims across families need a multiplicity correction.** Any statement made
  jointly about several families must apply **Holm–Bonferroni** over the families it
  covers, at a family-wise one-sided level of 2.5%. Examples: "C11 holds on the multimodal
  families", "on k of 5 families", or a pooled summary.
  - Each family's IUT p-value is the largest one-sided bootstrap p-value among its 36
    comparisons.
  - A family counts toward the joint statement only if it passes at its Holm-adjusted
    level.
  - Without that correction, the per-family statuses are reported side by side and no
    joint claim is made.
- **C11 status per family:**
  - **Supported:** for every baseline, budget and matching, the 95% interval of `d` lies
    entirely below 0.
  - **Not supported:** for some baseline, budget and matching, the interval lies entirely
    above 0 (that baseline is better).
  - **Inconclusive:** otherwise.
- **C12 status per family:** the same rules with the sign reversed. Supported requires
  every lower bound > 0; not supported requires some upper bound < 0. Degenerate cases
  (all rates 0 or all 1) are inconclusive.
- Each family's status is recorded separately. OOD results are reported (C8 conventions)
  and do not change the status.
- **Power** (meaningful effects approved: 0.25 decades of AUC; 10 percentage points of
  basin rate):
  - **Rule:** each comparison is powered at 80% for the threshold effect at the 95%
    two-sided level. Required paired instances `n = ((z_0.975 + z_0.8)·SD_d / δ)²`, with
    `z_0.975 + z_0.8 = 2.80`.
  - Under the IUT, the claim's power is at most the weakest comparison's, so `n` is set by
    the comparison with the largest `SD_d`. That is usually the closest baseline.
  - **A priori (normal approximation, assumed SDs; no data yet).** At n = 200 the minimal
    detectable effect is 0.198·SD_d. The 99% rule of the first draft needed 0.242·SD_d.

    | Effect δ | SD_d | n needed (95%) | n needed (99%, superseded) | MDE at n = 200 (95%) |
    |---|---|---|---|---|
    | 0.25 decades | 0.5 | 32 | 47 | 0.099 |
    | 0.25 decades | 1.0 | 126 | 187 | 0.198 |
    | 0.25 decades | 1.5 | 283 | 421 | 0.297 |
    | 0.25 decades | 2.0 | 503 | 748 | 0.396 |
    | 10 points | 0.2 | 32 | 47 | 0.040 |
    | 10 points | 0.3 | 71 | 106 | 0.059 |
    | 10 points | 0.5 | 197 | 292 | 0.099 |
    | 10 points | 0.7 | 385 | 573 | 0.139 |

    So 200 test instances suffice when `SD_d ≤ 1.26` decades (C11) or `SD_d ≤ 0.50`
    (C12).
  - **The binding check uses data:** at B-S2/B-S3, `SD_d` is estimated per family,
    comparison and budget from validation (baselines plus a pilot policy where available).
    The pre-registration sets each family's test size to the largest required `n`. Test
    sizes are raised, never the criteria loosened. If a family needs more than 1,000
    instances, that is reported before freezing.
- **Also reported, not criteria:** per-seed results, including the worst seed; failure and
  nonfinite counts; the table of all 36 per-family intervals.

## 7. Training

- **Method: antithetic evolution strategies** (Salimans et al. 2017) with centred-rank
  fitness shaping and Adam (optax, already in the `train` extra) on the ES gradient
  estimate.
  - ES needs no backpropagation through unrolls, so there are **no second-order unrolls**.
    It also accepts the non-differentiable, clipped metric directly.
  - PPO is a **pre-declared fallback** (decision 2), triggered only by the validation
    rule below.
- **Reward: clipped best-so-far regret at every unit.**
  - Per unit, `r_t = −L_i(t)` (§5.1), so an episode's mean reward is `−AUC_i(B)`, the
    evaluation metric itself.
  - POP's per-step form `max(0, f_best(t−1) − f(x_t))` telescopes to the total improvement
    and ignores *when* progress happens. The per-unit clipped regret keeps the anytime
    character and is clipped at the same floor and cap.
- **Data:** each target family's **training split only**.
  - Synthetic `f* = 0` is known.
  - Families 1 and 2b use the training-target references of the main plan (§10: 16
    starts × 1,000 units, owned).
  - Validation is used only for selection (best checkpoint every 25 generations) and early
    stopping. Test and OOD splits stay closed until the single final evaluation.
- **Settings:**
  - population P = 64 (32 antithetic pairs), ES noise 0.02;
  - M = 16 training instances per generation, common to the population;
  - budget sampled per episode from {20, 50, 100};
  - G ≤ 500 generations;
  - one policy per family; 5 seeds.

### 7.1 Pre-registered PPO fallback (validation only)

- **Validation checkpoints:** every 25 generations, each ES seed's current policy is
  evaluated on the family's validation split. The tracked score is the mean `AUC(B)` over
  B ∈ {20, 50, 100} under unit matching.
- **Plateau:** a seed has plateaued when either
  - **at least 200 generations** have run, and its best validation score improved by
    **less than 0.02 decades** over the last **4 checkpoints** (100 generations); or
  - it reaches G = 500.

  A family's ES training has plateaued when at least 3 of its 5 seeds have.
- **"Beats quasi-random search and BB on validation":** the seed-mean policy (best
  validation checkpoint per seed) has a lower mean validation `AUC(B)` than **both**
  quasi-random search and BB at **every** B ∈ {20, 50, 100}, under unit **and**
  wall-clock matching. These are point estimates on validation; no intervals are used for
  this switch.
- **Trigger:** PPO runs for a family if its ES training has plateaued **and** the ES policy
  does not beat quasi-random search and BB on validation. The trigger is evaluated once,
  when the family's ES runs stop.
- **PPO arm (pre-declared now):**
  - the same policy network as actor, plus a value head;
  - clipped surrogate 0.2, GAE λ = 0.95, γ = 1 (finite horizon);
  - the same per-unit reward `−L_i(t)`, the same training split and validation selection;
  - 5 seeds;
  - no new dependencies (JAX and optax).
  - It differentiates through the policy network per step, never through the optimization
    trajectory, so there are no second-order unrolls.
- **Which arm goes to test:** exactly one policy arm per family. It is ES if PPO was not
  triggered, PPO if it was, fixed before any test data is opened. The other arm's
  validation results are reported. **Test data never informs this choice.**
- **Compute:** PPO counts against the same 24 CPU-hour cap. If the projection exceeds the
  cap, PPO's update count is cut first; seeds are never cut.

## 8. Compute estimate and cap

The per-unit costs below come from F0's probe (`results/f0-probe-v0.8.0.json`). The policy
forward cost is an estimate that the B0 probe must measure.

- **Per generation:** 1,024 episodes × up to 50 policy steps = 51,200 steps.
- **Synthetic families and Family 1:**
  - about 26 µs per step (about 20 µs policy forward plus 2 cheap evaluations);
  - ≈ 1.3 s per generation, ≈ 11 min per run, plus validation checks;
  - 4 families × 5 seeds ≈ **4 CPU-hours**.
- **Family 2b:**
  - about 220 µs per step (≈ 100 µs per unit for batched owned evaluation, × 2, plus
    the forward pass);
  - ≈ 11 s per generation, ≈ 1.6 h per run;
  - 5 seeds ≈ **8 CPU-hours**.
- **Baseline tuning, references for the new synthetic splits, wall-clock calibration and
  the single test evaluation:** ≈ 4 CPU-hours. Synthetic references are free (`f*` known).
  Family 2b references dominate.
- **Total ≈ 16 CPU-hours. Proposed cap: 24 CPU-hours.**
  - If the B0 probe projects more than the cap, the pre-registered reduction order is:
    G 500 → 300, then M 16 → 8, then dropping the d = 8 OOD split.
  - Seeds (≥ 5) and test sizes are never reduced.
- For scale, POP reports about 56 H100-hours of training. A CPU budget three orders of
  magnitude smaller is a **stated risk** to C11 and C12; a negative result under this
  budget is reported as such and not generalized to learned policies at scale.

## 9. Long runs

Every run expected to exceed 10 minutes follows CLAUDE.md rule 9:
- a named tmux session (for example `neuralopt-v08b-train`);
- a git-ignored `runs/v08b-<stage>-<UTC stamp>/` directory holding:
  - `command.txt`: command, git SHA, `uv.lock` hash, UTC start;
  - `log.txt`;
  - an atomic `progress.json`: units (family × seed) done/total, generation, ETA;
  - checkpoints: one per finished unit via `scripts/run_checkpoint.py`, plus ES state every
    25 generations, so a resume loses at most 25 generations;
  - `exit_code`, and `DONE` only on success.
- `HANDOFF.md` is updated before and after each run.
- Versioned results go to `results/v0.8-b-*.json`. Frozen files are never overwritten.

## 10. Dependencies

| Need | Package | Licence | Pulls in | Where |
|---|---|---|---|---|
| CMA-ES | `cma` 4.x (pycma) | BSD-3-Clause | numpy | `benchmark` extra (+ dev group for tests) |
| TPE | `optuna` 5.x (`TPESampler`) | MIT | alembic, sqlalchemy, colorlog, tqdm, PyYAML, packaging | `benchmark` extra (+ dev group for tests) |
| ES, policy | none new | — | JAX, optax (existing `train` extra) | — |

- The PyPI metadata was checked on 2026-11-22 (cma 4.6.0, optuna 5.1.0).
- `hyperopt` (BSD-3; pulls scipy, networkx, cloudpickle) is the fallback for TPE if
  optuna's dependency footprint is unwanted.
- **These packages are never imported by `src/neuralopt`.** A test enforces this, and they
  are not added to the core dependencies. Tests that need them skip when they are absent,
  as the diffrax tests do.

## 11. Gates and stop points (track B)

| Stop | Content | Then |
|---|---|---|
| B-S1 | this plan | approved 2026-11-22 (PR #15) |
| B-S2 | the synthetic families with tests and hashed splits; baseline wrappers with call-accounting tests; B0 probe (timing, policy forward cost, projection against the cap); **headroom check with no learning** | **STOP and report** |
| B-S3 | pre-registration: grids, chosen configurations, frozen `C_a` curves, criteria, seeds, sizes; hashed | **STOP before any training** |
| B-S4 | training (tmux), validation-only selection, single test evaluation; register updated | **STOP and report**; the release decision is the user's |

- **B0 headroom rule (proposed):** a family is "not tested (B0: no headroom)" for both
  claims if the best baseline reaches the global basin with probability ≥ 0.95 at B = 20.
  Nothing would be left to win.
- B-S2 starts only after the main track's stop point 2 has been reported and this plan has
  been approved.

## 12. Risks

| Risk | Mitigation |
|---|---|
| Training scale far below POP | stated; negative results are scoped to this budget |
| Separability exploited | rotated Rastrigin and Ackley in distribution |
| Gradient price hidden by unit matching | wall-clock matching is a required condition |
| Python overhead (TPE, CMA-ES) dominates wall matching | reported per arm; unit matching is also required, so neither matching alone decides |
| Overfitting to a family | OOD splits reported; per-family claims only; no pooling |
| ES variance across seeds | ≥ 5 seeds; seeds resampled in the bootstrap; worst seed reported |
| Many comparisons | intersection-union within a family (95% per comparison); Holm–Bonferroni for any joint claim across families |
| ES fails to train | pre-declared PPO fallback on a validation-only trigger (§7.1) |
| `f*` not global (Families 1, 2b) | two-seed reproduction; updated when beaten |
| Context leaks the optimum | synthetic families get no context |
| The polish defines the basin | operational definition stated; zero-search sanity rows |

## 13. Prior work (verified; no novelty claim over it)

All of the following resolved exactly on OpenAlex on 2026-11-22. POP was also checked on
arxiv.org earlier.

- **Gärtner, E., Metz, L., Andriluka, M., Freeman, C. D., Sminchisescu, C.**
  *Transformer-Based Learned Optimization* (Optimus). CVPR 2023.
  doi:10.1109/CVPR52729.2023.01152.
- **Kobiolka, J., Frey, C., Shala, G., Kadra, A., Bedalli, E., Grabocka, J.** *POP:
  Prior-Fitted First-Order Optimization Policies.* arXiv:2602.15473, 2026. A preprint
  ("Under Review"), not peer-reviewed.
- **Goldie, A. D., Lu, C., Jackson, M. T., Whiteson, S., Foerster, J.** *Can Learned
  Optimization Make Reinforcement Learning Less Difficult?* (OPEN). NeurIPS 2024.
  doi:10.52202/079017-0177; arXiv:2407.07082.
- **Moudgil, A., Knyazev, B., Lajoie, G., Belilovsky, E.** *Celo: Training Versatile
  Learned Optimizers on a Compute Diet.* TMLR 2025. arXiv:2501.12670.
- **Meta-generalization caveats:**
  - **Metz, L., Freeman, C. D., Harrison, J., Maheswaranathan, N., Sohl-Dickstein, J.**
    *Practical tradeoffs between memory, compute, and performance in learned optimizers.*
    arXiv:2203.11860, 2022 (arXiv only).
  - **Rezk, F., Antoniou, A., Gouk, H., Hospedales, T. M.** *Is Scaling Learned Optimizers
    Worth It? Evaluating The Value of VeLO's 4000 TPU Months.* arXiv:2310.18191, 2023.
- **Methods used here:**
  - **Salimans, T., Ho, J., Chen, X., Sidor, S., Sutskever, I.** *Evolution Strategies as
    a Scalable Alternative to Reinforcement Learning.* arXiv:1703.03864, 2017.
  - **Hansen, N., Ostermeier, A.** *Completely Derandomized Self-Adaptation in Evolution
    Strategies.* Evolutionary Computation 9(2):159–195, 2001. doi:10.1162/106365601750190398.
  - **Bergstra, J., Bardenet, R., Bengio, Y., Kégl, B.** *Algorithms for Hyper-Parameter
    Optimization.* NeurIPS 2011.
  - **Berger, R. L.** *Multiparameter Hypothesis Testing and Acceptance Sampling.*
    Technometrics 24(4):295–300, 1982. doi:10.1080/00401706.1982.10487790.

## 14. Decisions (approved 2026-11-22)

1. Quasi-random search is in the "every baseline" set (§3.2).
2. ES first, PPO deferred, with a pre-registered PPO fallback triggered on validation only
   (§7.1).
3. A 24 CPU-hour cap with the stated reduction order (§8).
4. `optuna` for TPE, in the benchmark extra only (§10).
5. Power thresholds: 0.25 decades of AUC; 10 percentage points of basin rate (§6).
6. **Change:** C11 and C12 are decided per family as intersection-union tests with
   two-sided 95% intervals per comparison. A multiplicity correction (Holm–Bonferroni over
   families) applies only to claims made jointly across families. The power check was
   redone under this rule (§6).

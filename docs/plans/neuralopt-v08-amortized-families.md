# NeuralOpt v0.8 plan: amortized optimization over problem families

**Status: approved direction (Option A, 2026-11-22); this revision awaits approval.**
Nothing in it has been built. The claims it tests are listed, with their decision
criteria, in [`docs/research/v0.8-claims.md`](../research/v0.8-claims.md).

**Release freeze.**
- There is no v0.8.0 tag, PyPI upload or GitHub release until every claim in the register
  has a pre-registered result and the user has approved the release.
- Until a claim's status is "supported", the README, docs, CHANGELOG and release notes
  call it "under test".

## 1. Goal

Many problems come as a **family** `f(θ; c)`:
- θ is the parameter vector to estimate, for example kinetic rate constants;
- `c` is the instance's **context**: its observed data and its conditions, such as dose.

Members of a family share structure, and their solutions move with `c`.

**Question.** Can a model trained on one family supply **starting points** from which a
local optimizer lands in the right basin more often and more cheaply than generic
strategies? The generic strategies are a default start, multistart, and a
nearest-neighbour lookup. And does any advantage survive distribution shift and pay back
its offline cost?

**What is learned and what is not:**
- **Learned:** the start predictor `g_φ(c)`.
- **Not learned:** the refinement, which is NeuralOpt's own classical, pure-JAX L-BFGS.
  This differs from v0.4–v0.7, where the learned part was the step rule and every learned
  geometry lost to a fixed diagonal or the BB step.
- **Kept:** the learned-step transformer stays in the package as a documented research
  mode. Those negative results remain authoritative for it.

**Prior expectation, stated honestly:**
- A nearest-neighbour start is often strong for smooth families.
- Multistart may already be cheap for small, well-identified problems.
- Gate F0 measures this before anything is learned.

## 2. Pipeline at inference

```
c ──► g_φ(c) ──► θ̂ (or K candidates + confidence) ──► owned L-BFGS ──► θ̂*
                                                          │
             low confidence or poor refined fit ──► fallback: owned multistart L-BFGS
```

- Every objective and gradient call goes through one counted budget, including fallback
  calls.
- The `g_φ` forward pass is not an objective evaluation. Its cold and warm time is
  measured and reported (claim C10).
- No SciPy, Optimistix or other external solver runs at inference.

## 3. Component 1: owned L-BFGS (classical, not learned)

**Purpose:** the refinement for every NeuralOpt arm, and the local solver inside NeuralOpt
multistart and fallback. It is documented as a classical method.

### Design (`neuralopt.lbfgs`, opt-in, pure JAX)

- **Memory and direction:** limited memory `m` (default 10). The two-loop recursion runs
  only on curvature pairs with `sᵀy > ε‖s‖‖y‖`; other pairs are skipped and the skip is
  counted.
  - Initial scaling `γ = sᵀy / yᵀy` from the newest pair.
- **Box bounds, by projection:**
  - The **free set** is the variables not at a bound whose gradient would not push them
    out of the box.
  - The direction is the L-BFGS direction restricted to the free set; bound variables
    take the projected-gradient component.
  - Trial points are `P_box(x + α d)`.
  - A backtracking Armijo condition is checked along the projection arc, with sufficient
    decrease measured against `gᵀ(P_box(x + α d) − x)`.
  - The direction is reset to the projected gradient when it is not a descent direction.
- **Stopping:**
  - the projected-gradient norm `‖P_box(x − g) − x‖_∞` falls below a tolerance;
  - the relative change in f falls below a tolerance;
  - or the budget is exhausted.
- **Accounting** (the same convention as the NeuralOpt loop):
  - each line-search trial costs **one objective call**;
  - the gradient is computed only at the accepted point (one gradient call);
  - a trial is proposed only while `used + 2 ≤ budget`;
  - nonfinite values reject the trial and are counted;
  - counters are returned per solve.
- **Execution:**
  - one `lax.while_loop` inside one `jit`;
  - `vmap` over a batch of `(c, x0)`, and over K starts for multistart;
  - under `vmap`, both `lax.cond` branches are computed, but calls are charged only when
    taken, as in `neuralopt.compiled`.
- **Precision:** float64 for all v0.8 work.

### Tests (written first)

1. **Quadratics:**
   - Unconstrained SPD quadratics, d ∈ {2, 8, 32}, κ ∈ {1, 1e2, 1e4}: converge to the
     analytic minimizer, with projected-gradient norm ≤ 1e-8 within the budget.
   - On a quadratic, the stored pairs satisfy `y = H s` to 1e-10.
2. **SciPy reference at matched budget**, on the v0.7 engine-benchmark set (32 rotated
   κ = 100 quadratics, d = 16, 100 units each, `results/engine-benchmark-v0.7.0.json`
   protocol) and on the v0.7 rotated curvature tasks:
   - the owned L-BFGS reaches regret ≤ 1e-6 on at least as many tasks as SciPy L-BFGS-B
     minus one;
   - its median log10 regret is within 1.0 decade of SciPy's;
   - its median units to 1e-6 are within 1.5× SciPy's.
   - SciPy's counts use the existing harness accounting: every SciPy call is one
     objective plus one gradient.
3. **Bounded problems:**
   - quadratics whose minimizer lies outside the box, on a face, an edge or a corner;
   - Rosenbrock in a box that clips the optimum.
   - Each must reach the projected KKT point: projected-gradient norm ≤ 1e-8, with active
     bounds identified correctly. Iterates never leave the box.
4. **Accounting:**
   - counted objective and gradient calls equal wrapped-callback counts;
   - the budget is never exceeded;
   - nonfinite trials are rejected and counted.
5. **Parity:**
   - float64 single versus batched (`vmap`) solves give identical trajectories and counts
     to 1e-12;
   - repeated solves with a new context of the same shape do not recompile.

## 4. Component 2: start predictor `g_φ` (the learned part)

- **Input:** the context as a **set of observation tokens**. Each token holds
  `(time, measured value, species/channel id, condition features such as dose,
  observation-noise level if known)`.
  - Instances may have different numbers and spacings of time points.
  - Masked self-attention runs over the tokens, then masked mean and attention pooling.
  - Condition variables shared by the instance enter as an extra token.
- **Output:**
  - θ̂ in **log-parameter space**, mapped into the box with a scaled `tanh` (always
    inside the bounds);
  - **option K:** K candidate starts plus a **confidence score**: a calibrated
    probability that refinement from the top candidate reaches success.
- **Training data:**
  - **simulated instances:** parameters sampled from the family prior, model simulated,
    realistic noise added;
  - **solved instances:** the reference optimum θ* of each training instance, from a
    harness-only reference multistart. Training uses no test instance.
- **Losses compared** (pre-registered; the comparison is claim C7):
  1. **regression** to the reference optimum: `‖ĝ − log θ*‖²` in normalized
     coordinates;
  2. **objective-based:** minimize `f(θ̂; c)`, optionally after `k` unrolled L-BFGS steps;
  3. for K starts, a **best-of-K** loss `min_k ℓ(θ̂_k)`, with a diversity check so the
     candidates do not collapse.
  - The confidence head is trained with a binary loss on "refinement from this start
    reached success", from training-split solves only.
- **Inference:**
  1. one forward pass;
  2. L-BFGS from the top candidate, or from all K candidates batched;
  3. **fallback** to owned multistart L-BFGS when the confidence is below a threshold
     `τ_c` **or** the refined fit fails a fit check.
  - The fit check is the refined objective above a family-specific threshold derived from
    the noise model, for example a reduced-χ² bound.
  - `τ_c` and the fit threshold are tuned on validation only. The fallback calls count in
    the same budget.
- **Checkpoints:** a versioned schema, `neuralopt-start/v0.8-v1`, that fails closed. Earlier
  checkpoint formats are unaffected.

## 5. Arms

All arms use the same instances, matched **objective + gradient units**, float64 and CPU.
Wall time is reported cold, warm and batched.

| # | Arm | Start | Refinement | Role |
|---|---|---|---|---|
| A0 | default start + owned L-BFGS | family default θ_d | owned L-BFGS | no amortization (C1 comparator) |
| A1 | default start + SciPy L-BFGS-B | θ_d | SciPy | harness reference |
| A2 | multistart SciPy L-BFGS-B | Sobol starts, count and split validation-tuned | SciPy | strongest generic reference (C2, C5) |
| A2o | multistart owned L-BFGS | K Sobol/random starts | owned L-BFGS | K-random comparator (C6); also the fallback |
| A3 | nearest-neighbour start + owned L-BFGS | θ* of the nearest training context | owned L-BFGS | lookup bar (C3) |
| A4 | **learned start + owned L-BFGS** | `g_φ(c)` | owned L-BFGS | **primary** (C1–C4, C9, C10) |
| A4K | learned K starts + owned L-BFGS (batched) | top-K of `g_φ` | owned L-BFGS | C6 |
| A4F | A4 with confidence check and fallback | `g_φ(c)` | owned L-BFGS + fallback | C5, C8 |
| A5 | learned start + SciPy L-BFGS-B | `g_φ(c)` | SciPy | harness only: separates the start from the refinement |

**Nearest neighbour:**
- Distance is measured on a context representation fixed before any test data is seen:
  - observations interpolated to a common time grid per species;
  - each time point standardized with training-split statistics;
  - condition features appended.
- Lookup time is reported. Lookups use no objective units.

## 6. Metrics

- **Reference optimum `f*`:** harness-only multistart L-BFGS-B with 64 Sobol starts ×
  2,000 units, plus the true-parameter start when it is known.
  - It must be reproduced by a second Sobol seed within ε on ≥ 99% of validation
    instances. If not, the budget is raised.
  - If any arm beats it, it is updated and that fact is reported.
- **Success:** `f(θ̂) − f* ≤ ε_abs + ε_rel·|f*|`, with ε fixed per family before F0.
  **Success is measured on the objective.**
- **Parameter error:** reported only together with an **identifiability check**: profile
  flatness, or the condition number and eigenvalues of the Fisher information at θ* (for
  least squares, `JᵀJ/σ²`).
  - Directions with flat profiles or tiny Fisher eigenvalues are reported as
    non-identified, not as errors.
- **Cost:**
  - objective, gradient and HVP units;
  - accepted steps;
  - wall time cold (including compilation), warm and batched;
  - `g_φ` and nearest-neighbour time, listed separately;
  - fallback rate and fallback units.
- **Time and units to success:** censored at the budget. Also the success rate as a
  function of budget.
- **Break-even (C4):** `N* = T_offline / (t_generic − t_amortized)`.
  - `T_offline` = simulating the training set + solving all training targets (reference
    multistart) + training (compilation and updates).
  - `t` = mean time to success at a matched success rate.
  - Reported as a number with an interval and the measurement protocol. "No break-even"
    when `t_amortized ≥ t_generic`.
- **Statistics:**
  - paired per-instance differences, positive = worse;
  - percentile-bootstrap 95% intervals over instances;
  - paired intervals for success-rate differences;
  - per-family and per-split tables with failure and nonfinite counts;
  - ≥ 3 training seeds for learned arms.

## 7. Splits (fixed, hashed, written once)

- Instance `i` of split `s` uses `seed = hash(family, variant, split, i)`.
- Each split has a manifest listing contexts and true parameters, with its sha256 recorded.
- No tuning, model selection or threshold is set on a test split.

| Split | Purpose | Size (initial) |
|---|---|---|
| train | simulated + solved instances for `g_φ`, the nearest-neighbour index, confidence labels | 2,000 |
| validation | all tuning (baselines, τ_c, fit thresholds, early stopping) and F0 | 200 |
| test (in-distribution) | final comparison, opened once | 200 |
| test (OOD) | final comparison, opened once | 200 per shift |

OOD shifts, each a separate named split:
1. a higher noise level;
2. true parameters outside the training range, but inside the optimization box;
3. different sampling times: count, spacing and sparsity;
4. Family 2 only: doses outside the training range.

## 8. Families

### Family 1: damped-sinusoid fitting (synthetic, known multimodality)

- **Model:** `y(t) = a · exp(−λt) · sin(2πωt + φ) + ε`, with θ = (a, λ, ω, φ), d = 4.
- **Context:** the observed series.
- **Objective:** the mean squared residual.
- **Why:** frequency aliasing and harmonics give local minima in ω, whose number grows with
  the sampling span. This multimodality is known and controllable, and the truth is known.

### Family 2: two-step Michaelis–Menten cascade (ODE, d = 4–5)

- **Model:** `S →(E1) I →(E2) P`, with θ = log(V1, K1, V2, K2), plus an optional
  degradation rate (d = 5).
- **Solver:** float64 fixed-step RK4 in JAX. The step is chosen so the error is ≤ 1e-8
  relative against a 16× finer reference over the validation box.
- **Objective:** weighted least squares over the observed species, with known noise σ.
- **Variants:**
  - **2a, clean:** dense sampling and low noise. This is the control.
  - **2b, realistic:**
    - sparse sampling times (5–8 points, irregular);
    - noise at 5–10% of the signal;
    - condition-dependent inputs: dose (initial S0, or a bolus into S at a stated time),
      varied per instance;
    - only P observed, plus I at a subset of the times.
  - Variant 2b is intended to produce **real local minima and near-flat directions**.
    F0 checks whether it actually does, by counting the distinct converged optima per
    instance across reference starts.

### Family 3: real dataset from the user's kinetic modelling work (later)

Only after Families 1 and 2 pass their gates, and under its own addendum: provenance, the
split by experiment or condition, noise model, and licence.

## 9. Gate F0: headroom, with no learning

**What runs:**
- Arms A0, A1, A2, A2o and A3 and the reference;
- on the **validation** split of each family and variant;
- at budgets of {25, 50, 100, 200, 400, 800} units.

**Headroom on two axes, per family and variant:**
- `B95_multi`: the smallest budget at which multistart L-BFGS-B (A2) reaches 95% success.
  The owned version, A2o, is reported alongside.
- `B95_NN`: the same for the nearest-neighbour start + owned L-BFGS (A3).
- `c_local`: the median units of a single local solve from the default start (A0) to its
  own stopping criterion.

**Stop rule (pre-registered before F0 runs):**
- A family or variant **stops** only if `B95_multi ≤ 2 · c_local`. Then multistart is
  essentially as cheap as one local solve, and there is no headroom to amortize.
- If `B95_NN ≤ 2 · c_local`, a lookup already amortizes the family. This is recorded, and
  claim C3 becomes the decisive test for that family.
- Also reported: the share of instances that **no** generic arm solves at 800 units (the
  hard set), and the number of distinct converged optima per instance.

**Timing probe first:**
- It measures warm, batched µs per objective and per gradient call for each family.
- The full F0 compute is projected from it.
- **Cap: 6 CPU-hours through F0.** If the projection exceeds it, I stop and report before
  running.

## 10. Training targets and offline cost

- **Evaluation reference:** 128k units per instance.
- **Training targets:** a cheaper reference, 16 starts × 1,000 units. It must match the
  evaluation reference within ε on ≥ 98% of validation instances; if not, its budget is
  raised.
- All of this cost is part of `T_offline` (C4).

## 11. Gates and stop points

| Stop | After | Content |
|---|---|---|
| 1 | this plan, CLAUDE.md update and claims register committed (all claims "unverified") | approval |
| 2 | owned L-BFGS + tests, families + splits, gate F0 | headroom report per family |
| 3 | the v0.8 pre-registration, frozen and hashed, with every claim's criterion | approval before any training |
| 4 | the study (training, in-distribution test, OOD test) | claims register updated; every outcome reported as found; release decision is the user's |

Exploratory results may be reported but cannot change a claim's status. Each stop is
preceded by the full local quality gate.

## 12. Rules

- Test first; float64; no uncounted evaluations; never form secants from un-evaluated
  states.
- Frozen artifacts are never overwritten. v0.8 results are versioned
  (`results/*-v0.8.0.json`).
- External solvers are references only. No PINNs or PDE collocation. Certification stays
  independent and model-free.
- Squash merges with conventional PR titles. Research PRs may be merged only when every CI
  check passes. No tags or releases.

## 13. Prior work (verified; no novelty claim over it)

- **Amos, B.** *Tutorial on Amortized Optimization.* Foundations and Trends in Machine
  Learning, 2023. arXiv:2202.00665.
  - The taxonomy used here: regression-based versus objective-based amortization, and
    fully amortized versus semi-amortized models.
- **Sambharya, R., Hall, G., Amos, B., Stellato, B.** *Learning to Warm-Start Fixed-Point
  Optimization Algorithms.* Journal of Machine Learning Research 25(166):1–46, 2024.
  arXiv:2309.07835.
  - A learned warm start followed by a fixed iteration scheme. v0.8's pipeline follows
    this pattern, using L-BFGS on small nonconvex parameter-estimation families. **It is
    not a new method.**
- **Zammit-Mangion, A., Sainsbury-Dale, M., Huser, R.** *Neural Methods for Amortized
  Inference.* Annual Review of Statistics and Its Application, 2025.
  doi:10.1146/annurev-statistics-112723-034123.
  - A survey of amortized parameter estimation with neural networks: the comparison point
    for `g_φ` as an amortized point estimator.
  - Chosen over Cranmer, Brehmer and Louppe (PNAS 2020), which is broader.

All three were checked against OpenAlex; the JMLR volume was checked on jmlr.org.

## 14. Risks

| Risk | Mitigation |
|---|---|
| No headroom (multistart is cheap) | F0 stop rule `B95_multi ≤ 2·c_local`; report plainly |
| A lookup does as well | A3 is a required bar; C3 says so plainly if it holds |
| Regression averages over modes | objective-based and best-of-K losses (C7, C6) |
| Overconfident starts silently miss optima | confidence + fit-check fallback, evaluated as C5 |
| The owned L-BFGS is weaker than SciPy | its acceptance tests (§3) precede F0; A5 separates start from refinement |
| The reference optimum is not global | two-seed reproduction; updated when beaten |
| Non-identifiable parameters | success on the objective; parameter error only with identifiability checks |
| ODE solver error | fine-step reference check |
| Leakage | validation-only tuning; hashed splits; tests opened once |
| Unfair wall time | cold, warm and batched times separate; compilation in cold; same float64 CPU |

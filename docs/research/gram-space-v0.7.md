# v0.7 Track B: Gram-space history model, gates 1 and 2

**Status (2026-11-22): gate 1 passes; gate 2 fails its pre-stated criterion.** Under the
round's stop rule, this means the shared loop rule, not the model, is the limit. No
pre-registration was written and no training was run.

## The model (B1; opt-in, schema `neuralopt-policy/v0.7-gram-v1`)

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

## Gate 1: algebraic, float64 (`tests/test_gram.py`)

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

## Gate 2: capacity, no training (`results/gram-gates-v0.7.0.json`)

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

### Reading

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

## tf-h8 anisotropy cap (the frozen v0.6 checkpoints)

- With `geometry_bound = 10` and `lowrank_scale = 0.5`, the diagonal ranges over
  [0.1, 10] and λmax(UUᵀ) < 0.25.
- So the cap is 100× along the coordinate axes and about **3.5× off-axis**.
- Along tf-h8 runs (3 seeds × rotated κ=1000 × d = 7/10/16), the bound
  cond(P) ≤ (diag_max + ‖U‖_F²)/diag_min reached a maximum of 19.9 and a median of 3.3.

## Review findings and dispositions

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

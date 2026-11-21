# NeuralOpt v0.6 onward — trajectory-aware curvature plan

**Status (2026-11-21):** G0–G2 implemented (opt-in `history-v1`, shared `preconditioned` loop); first pre-registered round run: H1 supported, H2 and H3 not supported, so G3 attention, G5 and G6 are not started — see `docs/research/trajectory-curvature-v0.6.md`. Original status: research proposal / Claude Code handoff. Repo `main` inspected 2026-11-21, version 0.5.0. Frozen v0.5 negative result remains canonical.

## Objective and novelty test
Can a small sequence-aware learned optimizer infer useful **cross-coordinate curvature** from past evaluated states and transfer that geometry to unseen rotated, ill-conditioned, noisy and higher-dimensional landscapes, beating a **validation-tuned fixed-diagonal** baseline on best-feasible regret per objective/gradient cost and time?

Candidate contribution is not 'transformer + BFGS': Optimus (2023) and L-SR1 (2026) already learn curvature updates. The hypothesis is **observation-efficient, uncertainty-aware curvature encoding with rigorous cross-task generalization and precision/cost accounting**, integrated into NeuralOpt's existing learned direct and stochastic trajectories.

## Architectural contract
- Reuse `features.py`, `model.py`, `optimizer.py`, `training.py`, `stochastic_geometry.py`.
- Keep `parameter_features` with 5 features per coordinate and legacy `ModelConfig` checkpoint path intact. Use opt-in versioned `history_mode`/schema; no silent shape-based migration.
- Minimal bounded ring buffer `K = 4` then `K = 8` of actually evaluated normalized `(u, g_u, f, mask)` observations. Stable JAX PyTree shapes and bounded O(Kd) memory.
- Derived: `s = Δu`, `y = Δg_u`, reliable `s^T y`, norms, gradient alignment, objective difference, reliability/conditioning, optional low-rank sketch. Initial 'history summary' model should be non-attention; only later permit cross-attention.
- `P = diag(d) + U U^T` stays SPD; negative secant curvature is diagnostic and masks/learned uncertainty, not an illegal inverse Hessian assertion.
- For optimization, incumbent must be verified; stochastic search state may move uphill. No classical optimizer dispatch in production inference.

## Stage gates

### G0 — Baseline & info audit
Reproduce v0.5 and map the code/data path; verify current 5-feature embedding, frozen checkpoint path, and evaluation accounting. Deliver risk list and test plan. **No code changes before this.**

### G1 — Synthetic secant truth
Build deterministic test fixtures, not a new solver: known diagonal/rotated SPD Hessians, indefinite quadratics, noisy gradients and mismatched coordinate scales. Verify `y = H_u s` with measured pairs and the correct normalization Jacobian. Measure rank(span(S)), unbiased/noisy sample behaviour. Stop if source states or gradients are incoherent.

### G2 — Minimal masked history feature
Add fixed-shape ring buffer and no-attention secant statistics, versioned checkpoint schema, and cost-aware inference. Tests: rejected candidate, repeated point, noisy pair, no-history identity, dimensions 2/4/8/10, JIT/VMAP/grad. Stop if legacy regressions fail or memory/time cost dominates without improved information.

### G3 — Trajectory-aware neural representation
Only after G2 evidence: small temporal attention / Set Transformer-style cross-attention over K valid pairs and per-coordinate tokens, with explicit time order and permutation equivariance of coordinates. Compare against matched MLP history summary. No model-width scale-up as first step.

### G4 — Curvature-regularized meta-training
Multi-horizon regret plus optional soft inverse-secant consistency, HVP identification/probe ablation and reliability-aware masking. Test correct low-rank geometry in normalized coordinates, stabilization, true gradients and held-out performance. Do not incorporate PINN collocation or self-scaled Broyden solver as an inference engine.

### G5 — Stochastic reuse (only if G3/G4 outperform fixed diagonal)
Feed trajectory curvature into learned mass/mobility for HMC, overdamped and BAOAB. Keep fluctuation–dissipation, metric convention and sampling/inference correctness; compare to fixed diagonal with matched gradient/integration cost. Keep FP weak-form as optional regularizer; don't claim exact Gibbs invariance under online adaptation.

### G6 — Function-space and certified-search compatibility
Existing field covariance and Fourier Nyquist tests remain. Apply resolution-aware secant features only in a small compatible field experiment, then test optional feasible-incumbent candidate injection. Interval lower bounds, pruning and independent replay remain authoritative. Do not attempt a new PDE solver.

### G7 — Publication-grade decision
Run pre-registered multi-seed held-out family/dimension comparisons; compute paired confidence intervals and total cost. Publish negative results and select simplest supported model. If no benefit, retain fixed diagonal, archive experimental trajectory model, and stop expansion.

## Suggested research hypotheses
- H1: history improves rotated/ill-conditioned regret over no-history matched model at same evaluation budget.
- H2: attention adds value **beyond** hand-computed secant statistics and an MLP.
- H3: learned geometry beats validation-tuned fixed diagonal on unseen families/dimensions under cost accounting.
- H4: curvature transfer to BAOAB or function spaces improves downstream best feasible objective without sacrificing correctness.

## Prohibited shortcuts
No PINNs; no benchmark leakage; no new BFGS inference fallback; no unreplayable certificate; no uncounted HVP/gradient calls; no overwriting existing artifacts; no oversized architecture first; no default API break; no sweeping release claims.
